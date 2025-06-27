import os
import time

import hydra
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from great.dataset.dataset import GLOPBenchmarkDataset
from great.utils.constants import BENCHMARKING_RL_CONFIGS_PATH
from great.utils.utils import (
    get_all_trained_models_configs,
    get_matching_config,
    get_model,
    get_model_file,
    set_hyperparams,
    set_seeds,
)
from search.baselines import (
    farthest_insertion_tsp,
    nearest_insertion_tsp,
    nearest_neighbor_tsp,
)

#### In this file, we test our GREAT models on the same test dataset that were used in GLOP: https://arxiv.org/pdf/2312.08224v2


def get_random_problems(batch_size, node_cnt, problem_gen_params):
    ### from https://github.com/henry-yeh/GLOP/blob/master/eval_atsp/ATSProblemDef.py

    ################################
    # "tmat" type
    ################################

    int_min = problem_gen_params["int_min"]
    int_max = problem_gen_params["int_max"]
    scaler = problem_gen_params["scaler"]

    problems = torch.randint(
        low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt)
    )
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (
            problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)
        ).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)


def compute_tour_length(dists, tour):
    res = 0
    for i in range(len(tour) - 1):
        res += dists[tour[i]][tour[i + 1]]
    res += dists[tour[-1]][tour[0]]
    return res


def save_tours(tours, elapsed_time, name):
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, name)
    with open(path, "w") as f:
        for tour in tours:
            line = ",".join(
                map(str, tour)
            )  # convert each int to str and join with spaces
            f.write(line + "\n")
        f.write(f"Elapsed time {elapsed_time}")


def load_tours(name, TSP=False):
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, name)
    with open(path, "r") as file:
        lines = file.readlines()
    elapsed_time = lines[-1]
    lines = lines[:-1]

    elapsed_time = float(elapsed_time.split()[-1])

    tours = list()
    for line in lines:
        tours.append([int(x) for x in line.split(",")])
        if not TSP:
            if tours[-1][-1] != 0:
                print(line)
                assert False
    return tours, elapsed_time


def load_hgs_tours(name):
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, name)
    with open(path, "r") as file:
        lines = file.readlines()

    tours = list()
    for line in lines:
        tours.append([int(x) for x in line.split(",")])
        if tours[-1][-1] != 0:
            print(line)
            assert False
    return tours


def tours_loadable(name):
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, name)
    return os.path.isfile(path)


def get_gurobi_tours(data_dist, tsp_size):
    sol_file_name = data_dist + "_" + str(tsp_size) + "_solution.txt"
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(root, "final_models/TSP_RL")

    # Open the file in write mode
    with open(os.path.join(root, sol_file_name), "r") as file:
        lines = file.readlines()

    tours = []

    for line in lines:
        _l = line.split()
        _l = [int(x) for x in _l]
        tours.append(_l)

    return tours


def is_valid_cvrp(demand, capacity, tour):
    assert set(tour) == set(range(len(demand))), "Solution invalid"
    assert tour[0] == 0, "Didn't start at the depot"

    curr_load = 0
    for node in tour:
        if node == 0:
            curr_load = 0  # reset since we are at depot
        else:
            curr_load += demand[node]
        assert curr_load <= capacity, "Vehicle overloaded"

    return True


def is_valid_op(tour, distances, max_length):
    length = compute_tour_length(distances, tour)
    assert length <= max_length + 0.001, f"length: {length}, allowed max: {max_length}"
    assert tour[0] == 0, "didn't start at depot"
    assert tour[-1] == 0, "didn't end at depot"
    tour_with_out_depot = [x for x in tour if x != 0]
    assert len(set(tour_with_out_depot)) == len(
        tour_with_out_depot
    )  ### a non-depot node has been visited more than once

    # Remove leading zeros
    i = 0
    while i < len(tour) and tour[i] == 0:
        i += 1

    # Remove trailing zeros
    j = len(tour) - 1
    while j >= 0 and tour[j] == 0:
        j -= 1

    # Check the middle part for zeros
    for k in range(i, j + 1):
        if tour[k] == 0:
            assert False, "Depot was visited several times"

    return True


def compute_collected_prize(tour, prizes):
    curr_prize = 0
    for node in tour:
        if node != 0:
            curr_prize += prizes[node]

    return curr_prize


def get_glop_dists():
    ## adjusted from https://github.com/henry-yeh/GLOP/blob/master/eval_atsp/ATSProblemDef.py
    problem_gen_params = {"int_min": 0, "int_max": 1000 * 1000, "scaler": 1000 * 1000}

    torch.manual_seed(1234)
    problems_dict = dict()

    dataset_size = 30
    for scale in [150, 250, 1000]:
        problems = []
        for _ in tqdm(range(dataset_size)):
            problem = get_random_problems(1, scale, problem_gen_params)
            problems.append(problem)
        problems_dict[scale] = torch.cat(problems, dim=0)
    return problems_dict


@hydra.main(
    version_base=None,
    config_path=BENCHMARKING_RL_CONFIGS_PATH,
    config_name="TMAT_TSP_NF",
)
def eval(cfg: DictConfig):
    """
    This script is to evaluate an existing model in the RL task to find valid routing tours
    """
    glop_dists = get_glop_dists()
    glop_tour_dists = {
        150: 1.89,
        250: 2.04,
        1000: 2.33,
    }

    ### setting seeds
    set_seeds(1234)

    models_to_consider = [None, 200, 500]

    ### set hyperparameters
    params = set_hyperparams(cfg)
    assert params["task"] == "RL"

    ### init model
    model = get_model(params)
    model.to(params["device"])  # move to device
    model.eval()

    # get the config that matched a model that has been trained already and get its time_stamp (corresponds to an ID)
    configs = get_all_trained_models_configs()
    conf = get_matching_config(configs, params)
    model_time_stamp = conf["timestamp"]
    print(f"Evaluating trained model with timestamp: {model_time_stamp}")
    orig_timestamp = model_time_stamp

    for CL_instance_size in glop_dists:
        for model_to_consider in models_to_consider:
            print(
                f"Evaluating Model {orig_timestamp} finetuned for {model_to_consider} on VRP instances of size {CL_instance_size}"
            )

            # get the file where the model with the matching ID is stored and load it
            model_file = get_model_file(orig_timestamp, CL=model_to_consider)
            model.load_state_dict(torch.load(model_file, map_location=params["device"]))
            model.group_size = CL_instance_size  # set group size
            params["instance_size"] = CL_instance_size  # set instance size

            for dist in ["TAMT"]:
                params["data_distribution"] = dist
                model_time_stamp = orig_timestamp
                model_time_stamp += f"_CL_{CL_instance_size}"

                val_dataset = GLOPBenchmarkDataset(dists=glop_dists[CL_instance_size])
                # val_dataset = val_dataset[:100]  # for debugging
                val_loader = DataLoader(val_dataset, batch_size=1)
                print("Loaded dataset successfully!")

                # initialize list were generated tours are saved
                RL_tours = []
                RL_tours_1 = []
                nearest_insertion_tours = []
                farthest_insertion_tours = []
                nn_tours = []
                lkh_tours = []

                # evaluate the trained RL model using the predefined augmentation factor
                start_time = time.time()
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vdata.to(params["device"])
                        _, best_pi = model.get_tour(
                            vdata, augmentation_factor=params["augmentation_factor"]
                        )
                        if best_pi.dim() == 1:
                            best_pi = best_pi.unsqueeze(0)
                        RL_tours.extend(best_pi.cpu().numpy().tolist())
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                print(
                    f"Elapsed time {params['augmentation_factor']}x RL : {elapsed_time:.4f} seconds"
                )

                # evaluate the trained RL model using the no augmentation factor
                start_time = time.time()
                with torch.no_grad():
                    for i, vdata in enumerate(val_loader):
                        vdata.to(params["device"])
                        _, best_pi = model.get_tour(vdata, augmentation_factor=1)
                        if best_pi.dim() == 1:
                            best_pi = best_pi.unsqueeze(0)
                        RL_tours_1.extend(best_pi.cpu().numpy().tolist())
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                print(f"Elapsed time 1x RL : {elapsed_time:.4f} seconds")

                # extract distances from the data
                dists = val_dataset.dists.numpy()

                # compute tours using nearest insertion (this serves as a baseline)
                if params["problem"] == "TSP":
                    start_time = time.time()
                    nearest_insertion_tours = Parallel(n_jobs=16, verbose=10)(
                        delayed(nearest_insertion_tsp)(dist) for dist in dists
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(
                        f"Elapsed time nearest insertion : {elapsed_time:.4f} seconds"
                    )

                if params["problem"] == "TSP":
                    # compute tours using fartherst insertion (this serves as a baseline)
                    start_time = time.time()
                    farthest_insertion_tours = Parallel(n_jobs=16, verbose=10)(
                        delayed(farthest_insertion_tsp)(dist) for dist in dists
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(
                        f"Elapsed time farthest insertion : {elapsed_time:.4f} seconds"
                    )

                if params["problem"] == "TSP":
                    # compute tours using nearest neighbor (this serves as a baseline)
                    start_time = time.time()
                    nn_tours = Parallel(n_jobs=16, verbose=10)(
                        delayed(nearest_neighbor_tsp)(dist) for dist in dists
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Elapsed time nearest neighbor : {elapsed_time:.4f} seconds")

                # compute tours using LKH (this serves as a baseline)
                from great.utils.tour_wrapper import (  # get_EA4OP_results_op,
                    get_lkh_results_dists,
                )

                start_time = time.time()
                if params["problem"] == "TSP":
                    lkh_tours = Parallel(n_jobs=10, verbose=10)(
                        delayed(get_lkh_results_dists)(dist, not params["asymmetric"])
                        for dist in dists
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                if params["problem"] == "TSP":
                    print(f"Elapsed time LKH : {elapsed_time:.4f} seconds")

                # transform all tours into distances to compute optimality gaps
                print(
                    "Computing distances of algorithms for "
                    + params["data_distribution"]
                )
                rl_dists = []
                rl_dists_1 = []
                ni_dists = []
                fi_dists = []
                nn_dists = []
                lkh_dists = []
                for i in range(len(dists)):
                    d = dists[i]
                    rl = RL_tours[i]
                    rl_1 = RL_tours_1[i]
                    if params["problem"] == "TSP":
                        lkh = lkh_tours[i]
                        ni = nearest_insertion_tours[i]
                        fi = farthest_insertion_tours[i]
                        nn = nn_tours[i]

                    if params["problem"] == "TSP":  # check TSP solution validity
                        assert set(lkh) == set(
                            range(params["instance_size"])
                        ), "LKH solution invalid"
                        assert set(rl) == set(
                            range(params["instance_size"])
                        ), "RL solution invalid"
                        assert set(rl_1) == set(
                            range(params["instance_size"])
                        ), "RL x1 solution invalid"
                        assert set(ni) == set(
                            range(params["instance_size"])
                        ), "NI solution invalid"
                        assert set(fi) == set(
                            range(params["instance_size"])
                        ), "FI solution invalid"
                        assert set(nn) == set(
                            range(params["instance_size"])
                        ), "NN solution invalid"

                    if params["problem"] in ["TSP", "CVRP"]:
                        rl_dists.append(compute_tour_length(d, rl))
                        rl_dists_1.append(compute_tour_length(d, rl_1))

                    if params["problem"] == "TSP":
                        lkh_dists.append(compute_tour_length(d, lkh))
                        ni_dists.append(compute_tour_length(d, ni))
                        fi_dists.append(compute_tour_length(d, fi))
                        nn_dists.append(compute_tour_length(d, nn))

                rl_dists = round(sum(rl_dists) / len(rl_dists), 10)
                rl_dists_1 = round(sum(rl_dists_1) / len(rl_dists_1), 10)

                if params["problem"] == "TSP":
                    lkh_dists = round(sum(lkh_dists) / len(lkh_dists), 10)
                    ni_dists = round(sum(ni_dists) / len(ni_dists), 10)
                    fi_dists = round(sum(fi_dists) / len(fi_dists), 10)
                    nn_dists = round(sum(nn_dists) / len(nn_dists), 10)

                if params["problem"] in ["TSP", "CVRP"]:
                    print("Average length RL: " + str(rl_dists))
                    print("Average length RL x1: " + str(rl_dists_1))

                if params["problem"] == "TSP":
                    print("Average length LKH: " + str(lkh_dists))
                    print("Average length nearest insertion: " + str(ni_dists))
                    print("Average length farthest insertion: " + str(fi_dists))
                    print("Average length nearest neighbor: " + str(nn_dists))

                if params["problem"] == "TSP":
                    print("Gaps relative to LKH: ")
                    rl_gap = round(((rl_dists / lkh_dists) - 1) * 100, 2)
                    rl_1_gap = round(((rl_dists_1 / lkh_dists) - 1) * 100, 2)
                    ni_gap = round(((ni_dists / lkh_dists) - 1) * 100, 2)
                    fi_gap = round(((fi_dists / lkh_dists) - 1) * 100, 2)
                    nn_gap = round(((nn_dists / lkh_dists) - 1) * 100, 2)
                    glop_gap = round(
                        ((glop_tour_dists[CL_instance_size] / lkh_dists) - 1) * 100, 2
                    )

                print("Average gap RL: " + str(rl_gap) + "%")
                print("Average gap RL x1: " + str(rl_1_gap) + "%")
                if params["problem"] == "TSP":
                    print("Average gap nearest insertion: " + str(ni_gap) + "%")
                    print("Average gap farthest insertion: " + str(fi_gap) + "%")
                    print("Average gap nearest neighbor: " + str(nn_gap) + "%")
                    print("Gap GLOP: " + str(glop_gap) + "%")

                print("--------------------------------------------------------")


if __name__ == "__main__":
    eval()
