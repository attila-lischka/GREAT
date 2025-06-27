import os
import time

import hydra
import numpy as np
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

from great.dataset.dataset import GREATRLDataset
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
    greedy_ratio_op,
    nearest_insertion_tsp,
    nearest_neighbor_tsp,
)

### In this file, we benchmark the GREAT models that were fine tuned in the curriculum learning setting


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


@hydra.main(
    version_base=None,
    config_path=BENCHMARKING_RL_CONFIGS_PATH,
    config_name="EUC_TSP_NF",
)
def eval(cfg: DictConfig):
    """
    This script is to evaluate an existing model in the RL task to find valid routing tours
    """
    ### setting seeds
    set_seeds(1234)

    CL_instance_sizes = [200, 500]
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

    for CL_instance_size in CL_instance_sizes:
        for model_to_consider in models_to_consider:
            print(
                f"Evaluating Model {orig_timestamp} finetuned for {model_to_consider} on VRP instances of size {CL_instance_size}"
            )

            # get the file where the model with the matching ID is stored and load it
            model_file = get_model_file(orig_timestamp, CL=model_to_consider)
            model.load_state_dict(torch.load(model_file, map_location=params["device"]))
            model.group_size = CL_instance_size  # set group size
            params["instance_size"] = CL_instance_size  # set instance size

            if (
                params["data_distribution"] == "MIX"
            ):  # MIX is tested on all dists, otherwise only the trained dist
                benchmark_distributions = ["EUC", "TMAT", "XASY"]
            else:
                benchmark_distributions = [params["data_distribution"]]

            for dist in benchmark_distributions:
                params["data_distribution"] = dist
                if (
                    len(benchmark_distributions) > 1
                ):  # If we test the model for more than one dist, we need to specify when saving the solutions
                    model_time_stamp = orig_timestamp
                    model_time_stamp += "_MIX_"
                    model_time_stamp += dist
                else:
                    model_time_stamp = orig_timestamp
                model_time_stamp += f"_CL_{CL_instance_size}"

                val_dataset = GREATRLDataset(
                    data_dist=params["data_distribution"],
                    dataset_size=128,
                    problem=params["problem"],
                    instance_size=params["instance_size"],
                    seed=424242,
                    save_raw_data=True,
                    save_processed_data=True,
                )
                # val_dataset = val_dataset[:100]  # for debugging
                val_loader = DataLoader(val_dataset, batch_size=2)
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
                dists = []
                demands = []
                prizes = []
                for i, data in enumerate(val_loader):
                    if (
                        data.edge_attr.dim() == 2
                    ):  # augmented edge attributes --> extract distances only
                        d = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                            :, :, :, 0
                        ]
                    else:  # only 1D attributes
                        d = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
                    dists.append(d.cpu().numpy())

                    if params["problem"] == "CVRP":
                        dem = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                            :, :, :, -1
                        ]  # B x (N+1) x (N+1)

                        dem = dem[
                            :, :1, :
                        ].squeeze()  # B x (N+1) # reduce from an edge to a node level
                        demands.append(dem.cpu().numpy())
                    elif params["problem"] == "OP":
                        prize = to_dense_adj(
                            data.edge_index, data.batch, data.edge_attr
                        )[:, :, :, -1]  # B x (N+1) x (N+1)
                        prize = prize[
                            :, :1, :
                        ].squeeze()  # B x (N+1) # reduce from an edge to a node level
                        prizes.append(prize.cpu().numpy())

                        max_length = data.instance_feature[0].item()

                dists = np.vstack(dists)
                if params["problem"] == "CVRP":
                    demands = np.vstack(demands)
                    CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}
                    # capacity of the vehicle in the CVRP instance
                    capacity = CAPACITIES[params["instance_size"]]
                    demands = demands * capacity
                    demands = demands.astype(int)
                elif params["problem"] == "OP":
                    prizes = np.vstack(prizes)

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
                    get_EA4OP_results_op,
                    get_hgs_results_dists_cvrp,
                    get_lkh_results_dists,
                )

                op_tours = list()
                start_time = time.time()
                if params["problem"] == "TSP":
                    lkh_tours = Parallel(n_jobs=10, verbose=10)(
                        delayed(get_lkh_results_dists)(dist, not params["asymmetric"])
                        for dist in dists
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                elif params["problem"] == "CVRP":
                    hgs_tours = Parallel(n_jobs=16, verbose=10)(
                        delayed(get_hgs_results_dists_cvrp)(
                            dist,
                            dems,
                            capacity,
                            not params["asymmetric"],
                            10,  # assign a time limit of 60 sec to each instance
                        )
                        for dist, dems in zip(dists, demands)
                    )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                elif params["problem"] == "OP":
                    if params["data_distribution"] == "EUC":
                        start_time = time.time()
                        ea4op_tours = Parallel(n_jobs=1, verbose=10)(
                            delayed(get_EA4OP_results_op)(
                                dist, prize, max_length, False
                            )
                            for dist, prize in zip(dists, prizes)
                        )
                        # Calculate the elapsed time
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Elapsed time EA4OP : {elapsed_time:.4f} seconds")

                    start_time = time.time()
                    for dist, prize in zip(dists, prizes):
                        op_tours.append(
                            greedy_ratio_op(
                                dist,
                                prize,
                                max_length,
                                params["data_distribution"] == "XASY",
                            )
                        )
                    # Calculate the elapsed time
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                if params["problem"] == "TSP":
                    print(f"Elapsed time LKH : {elapsed_time:.4f} seconds")
                elif params["problem"] == "CVRP":
                    print(f"Elapsed time HGS : {elapsed_time:.4f} seconds")
                elif params["problem"] == "OP":
                    print(f"Elapsed time Greedy-OP : {elapsed_time:.4f} seconds")

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
                hgs_dists = []
                op_dists = []
                ea4op_dists = []
                for i in range(len(dists)):
                    d = dists[i]
                    rl = RL_tours[i]
                    rl_1 = RL_tours_1[i]
                    if params["problem"] == "TSP":
                        lkh = lkh_tours[i]
                        ni = nearest_insertion_tours[i]
                        fi = farthest_insertion_tours[i]
                        nn = nn_tours[i]
                    elif params["problem"] == "CVRP":
                        hgs = hgs_tours[i]
                    elif params["problem"] == "OP":
                        op = op_tours[i]
                        if params["data_distribution"] == "EUC":
                            ea4op = ea4op_tours[i]

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
                    elif params["problem"] == "CVRP":  # check CVRP solution is valid
                        assert is_valid_cvrp(demands[i], capacity, rl)
                        assert is_valid_cvrp(demands[i], capacity, rl_1)
                        assert is_valid_cvrp(demands[i], capacity, hgs)
                    elif params["problem"] == "OP":  # check OP slution is valid
                        assert is_valid_op(rl, d, max_length)
                        assert is_valid_op(rl_1, d, max_length)
                        assert is_valid_op(op, d, max_length)
                        if params["data_distribution"] == "EUC":
                            assert is_valid_op(ea4op, d, max_length)

                    if params["problem"] in ["TSP", "CVRP"]:
                        rl_dists.append(compute_tour_length(d, rl))
                        rl_dists_1.append(compute_tour_length(d, rl_1))
                    elif params["problem"] == "OP":
                        rl_dists.append(compute_collected_prize(rl, prizes[i]))
                        rl_dists_1.append(compute_collected_prize(rl_1, prizes[i]))
                        op_dists.append(compute_collected_prize(op, prizes[i]))
                        if params["data_distribution"] == "EUC":
                            ea4op_dists.append(
                                compute_collected_prize(ea4op, prizes[i])
                            )

                    if params["problem"] == "TSP":
                        lkh_dists.append(compute_tour_length(d, lkh))
                        ni_dists.append(compute_tour_length(d, ni))
                        fi_dists.append(compute_tour_length(d, fi))
                        nn_dists.append(compute_tour_length(d, nn))

                    elif params["problem"] == "CVRP":
                        hgs_dists.append(compute_tour_length(d, hgs))

                rl_dists = round(sum(rl_dists) / len(rl_dists), 10)
                rl_dists_1 = round(sum(rl_dists_1) / len(rl_dists_1), 10)

                if params["problem"] == "TSP":
                    lkh_dists = round(sum(lkh_dists) / len(lkh_dists), 10)
                    ni_dists = round(sum(ni_dists) / len(ni_dists), 10)
                    fi_dists = round(sum(fi_dists) / len(fi_dists), 10)
                    nn_dists = round(sum(nn_dists) / len(nn_dists), 10)
                elif params["problem"] == "CVRP":
                    hgs_dists = round(sum(hgs_dists) / len(hgs_dists), 10)
                elif params["problem"] == "OP":
                    op_dists = round(sum(op_dists) / len(op_dists), 10)
                    if params["data_distribution"] == "EUC":
                        ea4op_dists = round(sum(ea4op_dists) / len(ea4op_dists), 10)

                if params["problem"] in ["TSP", "CVRP"]:
                    print("Average length RL: " + str(rl_dists))
                    print("Average length RL x1: " + str(rl_dists_1))
                elif params["problem"] == "OP":
                    print("Average prize RL: " + str(rl_dists))
                    print("Average prize RL x1: " + str(rl_dists_1))
                    print("Average prize greedy: " + str(op_dists))
                    if params["data_distribution"] == "EUC":
                        print("Average prize EA4OP: " + str(ea4op_dists))

                if params["problem"] == "TSP":
                    print("Average length LKH: " + str(lkh_dists))
                    print("Average length nearest insertion: " + str(ni_dists))
                    print("Average length farthest insertion: " + str(fi_dists))
                    print("Average length nearest neighbor: " + str(nn_dists))
                elif params["problem"] == "CVRP":
                    print("Average length HGS: " + str(hgs_dists))

                if params["problem"] == "TSP":
                    print("Gaps relative to LKH: ")
                    rl_gap = round(((rl_dists / lkh_dists) - 1) * 100, 2)
                    rl_1_gap = round(((rl_dists_1 / lkh_dists) - 1) * 100, 2)
                    ni_gap = round(((ni_dists / lkh_dists) - 1) * 100, 2)
                    fi_gap = round(((fi_dists / lkh_dists) - 1) * 100, 2)
                    nn_gap = round(((nn_dists / lkh_dists) - 1) * 100, 2)
                elif params["problem"] == "CVRP":
                    print("Gaps relative to HGS: ")
                    rl_gap = round(((rl_dists / hgs_dists) - 1) * 100, 2)
                    rl_1_gap = round(((rl_dists_1 / hgs_dists) - 1) * 100, 2)
                elif params["problem"] == "OP":
                    print("Gaps relative to Greedy: ")

                    rl_gap = round(((op_dists / rl_dists) - 1) * 100, 2)
                    rl_1_gap = round(((op_dists / rl_dists_1) - 1) * 100, 2)
                    if params["data_distribution"] == "EUC":
                        ea4op_gap = round(((op_dists / ea4op_dists) - 1) * 100, 2)

                print("Average gap RL: " + str(rl_gap) + "%")
                print("Average gap RL x1: " + str(rl_1_gap) + "%")
                if params["problem"] == "TSP":
                    print("Average gap nearest insertion: " + str(ni_gap) + "%")
                    print("Average gap farthest insertion: " + str(fi_gap) + "%")
                    print("Average gap nearest neighbor: " + str(nn_gap) + "%")
                if params["problem"] == "OP":
                    print("Average gap EA4OP: " + str(ea4op_gap) + "%")

                print("--------------------------------------------------------")


if __name__ == "__main__":
    eval()
