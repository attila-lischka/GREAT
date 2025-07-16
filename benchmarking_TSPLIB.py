import gzip
import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.utils import to_dense_adj

from great.dataset.data_object import get_asymmetric_data_object, get_data_object
from great.utils.constants import BENCHMARKING_RL_CONFIGS_PATH, DATA_PATH
from great.utils.tour_wrapper import (  # get_EA4OP_results_op,
    get_hgs_results_dists_cvrp,
    get_lkh_results_dists,
)
from great.utils.utils import (
    get_all_trained_models_configs,
    get_matching_config,
    get_model,
    get_model_file,
    set_hyperparams,
    set_seeds,
)

### this file is for benchmarking and evaluating trained GREAT models on
### TSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
### ATSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
### CVRPLib http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ (instances with more than 200 nodes as well as XML100 are not considered)
### OPLib https://github.com/bcamath-ds/OPLib/tree/master

MAX_INSTANCE_SIZE = 100

atsp_instances = {  ## http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ATSP.html
    "br17": 39,
    "ft53": 6905,
    "ft70": 38673,
    "ftv33": 1286,
    "ftv35": 1473,
    "ftv38": 1530,
    "ftv44": 1613,
    "ftv47": 1776,
    "ftv55": 1608,
    "ftv64": 1839,
    "ftv70": 1950,
    "ftv90": 1579,
    "ftv100": 1788,
    "ftv110": 1958,
    "ftv120": 2166,
    "ftv130": 2307,
    "ftv140": 2420,
    "ftv150": 2611,
    "ftv160": 2683,
    "ftv170": 2755,
    "kro124p": 36230,
    "p43": 5620,
    "rbg323": 1326,
    "rbg358": 1163,
    "rbg403": 2465,
    "rbg443": 2720,
    "ry48p": 14422,
}

tsp_instances = {  ## http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    "a280": 2579,
    "ali535": 202339,
    "att48": 10628,
    "att532": 27686,
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    "brazil58": 25395,
    "brd14051": 469385,
    "brg180": 1950,
    "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    "dantzig42": 699,
    "dsj1000_EUC_2D": 18659688,
    "dsj1000_CEIL_2D": 18660188,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    "fri26": 937,
    "gil262": 2378,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "gr48": 5046,
    "gr96": 55209,
    "gr120": 6942,
    "gr137": 69853,
    "gr202": 40160,
    "gr229": 134602,
    "gr431": 171414,
    "gr666": 294358,
    "hk48": 11461,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728,
    "pla33810": 66048945,
    "pla85900": 142382641,
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    "si175": 21407,
    "si535": 48450,
    "si1032": 92650,
    "st70": 675,
    "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556,
}


def init_distances_fast(nodes):
    """
    Function that return a distance matrix based on the Euclidean distances of the passed nodes.
    nodes: coordinates of the nodes for which we want to compute the distances. Dimension: n * 2
    Returns: An n * n matrix where entry (i, j) corresponds to the Euclidean distance between node i and j.
    """
    nodes = np.array(nodes)
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances


def compute_tour_length(dists, tour):
    res = 0
    for i in range(len(tour) - 1):
        res += dists[tour[i]][tour[i + 1]]
    res += dists[tour[-1]][tour[0]]
    return res


def transform_strings_to_data(string_instances, instances_to_load, file_names=None):
    ### transforms a list of strings representing VRP data into PyG objects
    data_list = []
    if instances_to_load == "ATSPLIB":  ### ATSP instances are all in a similar format
        for instance in string_instances:
            name = instance[0]
            assert name[:4] == "NAME"
            name = name.split()[1]
            dimension = instance[3]
            assert dimension[:9] == "DIMENSION", dimension
            dimension = int(dimension.split()[1])
            if dimension > MAX_INSTANCE_SIZE:
                continue
            weights = []
            for line in instance[7:-1]:  # last element is "EOF" and we dont need that
                weights.extend(line.split())
            weights = [int(w) for w in weights]
            weights = np.array(weights).reshape(dimension, dimension)
            np.fill_diagonal(weights, 0)
            dists = weights / np.max(weights)
            data_obj = get_asymmetric_data_object(dists, None, None, None)
            data_list.append((name, data_obj, weights))

    elif (
        instances_to_load == "TSPLIB"
    ):  ### TSP instances are in different formats so code needs to be more dynamic
        for instance in string_instances:
            name = instance[0]
            assert name[:4] == "NAME"
            name = name.split()[-1]
            for line in instance:
                if "DIMENSION" in line:
                    dimension = line.split()[-1]
                    dimension = int(dimension)
                    break
            if dimension > MAX_INSTANCE_SIZE:
                continue
            for line in instance:
                if "EDGE_WEIGHT_TYPE" in line:
                    edge_type = line.split()[-1]
                    break
            if edge_type not in ["EUC_2D"]:
                continue  ### we do not support distances like "ATT" in att532 or "GEO" in gr137
                # while GREAT can process explicit (a)symmetric data (e.g. symmetric but no triangle inequality), we have not trained our EUC model on such data so we do not eval it either

            for i, line in enumerate(instance):
                if "NODE_COORD_SECTION" in line:
                    index = i
                    break
            weights = []
            for line in instance[index + 1 :]:
                if line[0].isdigit():
                    weights.extend(line.split()[1:])
                else:
                    break

            weights = [float(w) for w in weights]
            coords = np.array(weights).reshape(dimension, 2)
            weights = np.array(weights).reshape(dimension, 2)

            # Find smallest x and y
            min_x = np.min(weights[:, 0])
            min_y = np.min(weights[:, 1])

            # Subtract them from x and y columns respectively
            weights = weights - np.array([min_x, min_y])
            max_value = np.max(weights)
            weights /= max_value

            ### coordinates on exactly (0,0) are not allowed otherwise pointerformer augmentation crashed when computing the thetas
            weights = weights * (0.9909 - 0.0001) + 0.0001

            data_obj = get_data_object(weights, False, False)

            dists = init_distances_fast(coords)
            data_list.append((name, data_obj, dists))

    elif (
        instances_to_load == "CVRPLIB"
    ):  # we leave out the "XML100" data (too many instances)
        for instance in string_instances:
            name = instance[0]
            assert name[:4] == "NAME"
            name = name.split()[-1]
            for line in instance:
                if "DIMENSION" in line:
                    dimension = line.split()[-1]
                    dimension = int(dimension)
                    break
            if dimension > MAX_INSTANCE_SIZE:
                continue
            for line in instance:
                if "EDGE_WEIGHT_TYPE" in line:
                    edge_type = line.split()[-1]
                    break
            if edge_type not in ["EUC_2D"]:
                continue  ### we do not support distances like "ATT" in att532 or "GEO" in gr137
                # while GREAT can process explicit (a)symmetric data (e.g. symmetric but no triangle inequality), we have not trained our EUC model on such data so we do not eval it either

            for i, line in enumerate(instance):
                if "NODE_COORD_SECTION" in line:
                    index = i
                    break
            weights = []
            for line in instance[index + 1 :]:
                if line[0].isdigit():
                    weights.extend(line.split()[1:])
                else:
                    break

            weights = [float(w) for w in weights]
            coords = np.array(weights).reshape(dimension, 2)
            weights = np.array(weights).reshape(dimension, 2)

            # Find smallest x and y
            min_x = np.min(weights[:, 0])
            min_y = np.min(weights[:, 1])

            # Subtract them from x and y columns respectively
            weights = weights - np.array([min_x, min_y])
            max_value = np.max(weights)
            weights /= max_value
            coordinates = weights
            ### coordinates on exactly (0,0) are not allowed otherwise pointerformer augmentation crashed when computing the thetas
            coordinates = coordinates * (0.9909 - 0.0001) + 0.0001
            weights = init_distances_fast(weights)

            for i, line in enumerate(instance):
                if "DEPOT_SECTION" in line:
                    assert instance[i + 1] == "1" and instance[i + 2] == "-1"
                    break

            for i, line in enumerate(instance):
                if "DEMAND_SECTION" in line:
                    index = i
                    break
            demands = []
            for line in instance[index + 1 :]:
                if line[0].isdigit():
                    demands.extend(line.split()[1:])
                else:
                    break
            demands = [float(d) for d in demands]
            assert demands[0] == 0

            for line in instance:
                if "CAPACITY" in line:
                    capa = line.split()[-1]
                    capa = float(capa)
                    break
            demands_normalized = np.array(demands) / capa
            demands_normalized[0] = -1

            data_obj = get_asymmetric_data_object(
                weights, demands_normalized, capa, coordinates
            )

            dists = init_distances_fast(coords)
            data_list.append((name, data_obj, dists, demands, capa))
    elif instances_to_load == "OPLIB":
        for n, instance in zip(file_names, string_instances):
            name = os.path.basename(n)[:-6]
            for line in instance:
                if "DIMENSION" in line:
                    dimension = line.split()[-1]
                    dimension = int(dimension)
                    break
            if dimension > MAX_INSTANCE_SIZE:
                continue
            for line in instance:
                if "EDGE_WEIGHT_TYPE" in line:
                    edge_type = line.split()[-1]
                    break
            if edge_type not in ["EUC_2D"]:
                continue  ### we do not support distances like "ATT" in att532 or "GEO" in gr137
                # while GREAT can process explicit (a)symmetric data (e.g. symmetric but no triangle inequality), we have not trained our EUC model on such data so we do not eval it either

            for i, line in enumerate(instance):
                if "NODE_COORD_SECTION" in line:
                    index = i
                    break
            weights = []
            for line in instance[index + 1 :]:
                if line[0].isdigit():
                    weights.extend(line.split()[1:])
                else:
                    break

            weights = [float(w) for w in weights]
            coords = np.array(weights).reshape(dimension, 2)
            weights = np.array(weights).reshape(dimension, 2)

            # Find smallest x and y
            min_x = np.min(weights[:, 0])
            min_y = np.min(weights[:, 1])

            # Subtract them from x and y columns respectively
            weights = weights - np.array([min_x, min_y])
            max_value = np.max(weights)
            weights /= max_value
            coordinates = weights
            ### coordinates on exactly (0,0) are not allowed otherwise pointerformer augmentation crashed when computing the thetas
            coordinates = coordinates * (0.9909 - 0.0001) + 0.0001

            weights = init_distances_fast(weights)

            dists = init_distances_fast(coords)

            tour_orig = dists[0][1] + dists[1][2] + dists[2][0]
            tour_new = weights[0][1] + weights[1][2] + weights[2][0]
            scaling = tour_orig / tour_new

            for i, line in enumerate(instance):
                if "DEPOT_SECTION" in line:
                    assert instance[i + 1] == "1" and instance[i + 2] == "-1"
                    break

            for i, line in enumerate(instance):
                if "NODE_SCORE_SECTION" in line:
                    index = i
                    break
            prizes = []
            for line in instance[index + 1 :]:
                if line[0].isdigit():
                    prizes.extend(line.split()[1:])
                else:
                    break
            prizes = [float(d) for d in prizes]

            for line in instance:
                if "COST_LIMIT" in line:
                    max_dist = line.split()[-1]
                    max_dist_orig = float(max_dist)
                    max_dist = max_dist_orig / scaling
                    break

            max_val = max(prizes)
            prizes_normalized = np.array([x / max_val for x in prizes])
            prizes_normalized[0] = 0  # depot has no prize, it is visited anyway

            temp_distances = weights / max_dist

            node_features = prizes_normalized[..., np.newaxis]
            depot_distances = temp_distances[:, 0]
            depot_distances = depot_distances[..., np.newaxis]

            node_features = np.concatenate((depot_distances, node_features), axis=-1)

            distances_normalized = weights / max_dist
            weights = weights[..., np.newaxis]
            distances_normalized = distances_normalized[..., np.newaxis]
            weights = np.concatenate((weights, distances_normalized), axis=-1)

            data_obj = get_asymmetric_data_object(
                weights, node_features, max_dist, coordinates
            )

            data_list.append((name, data_obj, dists, prizes, max_dist_orig))

    return data_list


def get_problem_instances(instances_to_load):
    ### reads in TSPLIB files and calls other function to tansform them in to PyG objects

    if instances_to_load == "ATSPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_atsp")
        suffix = ".atsp.gz"
        instances = [
            os.path.abspath(os.path.join(load_path, f))
            for f in os.listdir(load_path)
            if os.path.isfile(os.path.join(load_path, f))
        ]
        instances = [f for f in instances if suffix in f]
    elif instances_to_load == "TSPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_tsp")
        suffix = ".tsp.gz"
        instances = [
            os.path.abspath(os.path.join(load_path, f))
            for f in os.listdir(load_path)
            if os.path.isfile(os.path.join(load_path, f))
        ]
        instances = [f for f in instances if suffix in f]
    elif instances_to_load == "CVRPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_VRP")
        instances = []
        for root, dirs, files in os.walk(load_path):
            for file in files:
                instances.append(os.path.join(root, file))
        instances = [f for f in instances if ".vrp" in f]
    elif instances_to_load == "OPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_OP")
        instances = []
        for root, dirs, files in os.walk(load_path):
            for file in files:
                instances.append(os.path.join(root, file))
        instances = [f for f in instances if ".oplib" in f]

    string_instances = []
    for file_path in instances:
        if ".gz" in file_path:
            with gzip.open(file_path, "rt") as f:  # 'rt' = read text mode
                instance = []
                for line in f:
                    instance.append(line.strip())
                string_instances.append(instance)
        else:
            with open(file_path, "r") as f:
                instance = []
                for line in f:
                    instance.append(line.strip())
                string_instances.append(instance)

    data_instances = transform_strings_to_data(
        string_instances, instances_to_load, instances
    )

    return data_instances


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


def get_best_known(name, instances_to_load):
    if instances_to_load == "ATSPLIB":
        return atsp_instances[name]
    elif instances_to_load == "TSPLIB":
        return tsp_instances[name]
    elif instances_to_load == "CVRPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_vrp")
        instances = []
        for root, dirs, files in os.walk(load_path):
            for file in files:
                instances.append(os.path.join(root, file))
        instances = [f for f in instances if ".sol" in f]
        instances = [f for f in instances if (name + ".sol").lower() in f.lower()]
        assert len(instances) == 1, f"could not find a solution for {name}"
        with open(instances[0], "r") as f:
            for line in f:
                if "Cost" in line or "cost" in line:
                    return float(line.split()[-1])

    elif instances_to_load == "OPLIB":
        load_path = os.path.join(DATA_PATH, "ALL_OP")
        instances = []
        for root, dirs, files in os.walk(load_path):
            for file in files:
                instances.append(os.path.join(root, file))
        instances = [f for f in instances if ".sol" in f]
        instances = [f for f in instances if (name + ".sol").lower() in f.lower()]
        assert len(instances) == 1, f"could not find a solution for {name}"
        with open(instances[0], "r") as f:
            for line in f:
                if "ROUTE_SCORE" in line:
                    return float(line.split()[-1])


def human_readible_model_name(params):
    name = params["data_distribution"] + "_" + params["problem"] + "_"

    if "matnet" in params and params["matnet"]:
        name += "matnet"
    elif "pointerformer" in params and params["pointerformer"]:
        name += "pointerformer"
    else:
        if params["nodeless"]:
            name += "NF"
        else:
            name += "NB"
    return name


def load_or_create_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                if not isinstance(data, dict):
                    raise ValueError("JSON content is not a dictionary.")
                return data
            except json.JSONDecodeError:
                print(
                    "Warning: File exists but contains invalid JSON. Returning empty dictionary."
                )
                return {}
    else:
        return {}


def save_result(
    model_name, instance_name, rl_gap, rl_gap_wrt_heur, heur_gap, heur_length, params
):
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, "libresults.json")
    results = load_or_create_json(path)
    if params["problem"] not in results:
        results[params["problem"]] = dict()

    if instance_name not in results[params["problem"]]:
        results[params["problem"]][instance_name] = dict()

    if model_name not in results[params["problem"]][instance_name]:
        results[params["problem"]][instance_name][model_name] = dict()

    results[params["problem"]][instance_name][model_name]["RL_gap_wrt_best"] = rl_gap
    results[params["problem"]][instance_name][model_name]["rl_gap_wrt_heur"] = (
        rl_gap_wrt_heur
    )
    results[params["problem"]][instance_name]["heuristic_gap"] = heur_gap
    results[params["problem"]][instance_name]["heuristic_length"] = heur_length

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


@hydra.main(
    version_base=None,
    config_path=BENCHMARKING_RL_CONFIGS_PATH,
    config_name="EUC_CVRP_NB",
)
def eval(cfg: DictConfig):
    """
    This script is to evaluate an existing model in the RL task to find valid routing tours
    """
    ### setting seeds
    set_seeds(1234)

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

    # get the file where the model with the matching ID is stored and load it
    model_file = get_model_file(model_time_stamp)
    model.load_state_dict(torch.load(model_file, map_location=params["device"]))
    model_name = human_readible_model_name(params)

    instances_to_load = None
    if params["problem"] == "TSP":
        if (
            params["data_distribution"]
            == "EUC"  # or params["data_distribution"] == "MIX"
        ):  # mix works for both, ATSPLIB and TSPLIB
            instances_to_load = "TSPLIB"
        else:
            instances_to_load = "ATSPLIB"
    elif params["problem"] == "CVRP":
        assert params["data_distribution"] in [
            "EUC",
            "MIX",
        ], "CVRP only has EUC real world benchmarking instances"
        instances_to_load = "CVRPLIB"
    elif params["problem"] == "OP":
        assert params["data_distribution"] in [
            "EUC",
            "MIX",
        ], "OP only has EUC real world benchmarking instances"
        instances_to_load = "OPLIB"
    else:
        raise NotImplementedError("Unknown Problem Type")

    data_objects = get_problem_instances(instances_to_load)

    for instance in data_objects:
        data = instance[1]
        if params["problem"] == "TSP":
            model.group_size = data.num_nodes  # set POMO size accordingly
        else:
            model.group_size = data.num_nodes - 1
        if data.num_nodes > MAX_INSTANCE_SIZE:
            print(f"Skipping instance {instance[0]} which has {data.num_nodes} cities.")
            continue

        if (
            data.edge_attr.dim() == 2
        ):  # augmented edge attributes --> extract distances only
            d = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[:, :, :, 0]
        else:  # only 1D attributes
            d = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
        d = d.squeeze(0).numpy()

        if params["problem"] == "CVRP":
            dem = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                :, :, :, -1
            ]  # B x (N+1) x (N+1)

            dem = dem[
                :, :1, :
            ].squeeze()  # B x (N+1) # reduce from an edge to a node level
            dem = dem.numpy()

        _, best_pi = model.get_tour(
            instance[1], augmentation_factor=params["augmentation_factor"]
        )
        if best_pi is not None:
            best_pi = best_pi.numpy().tolist()

        best_known_length = get_best_known(instance[0], instances_to_load)

        if params["problem"] == "TSP":
            lkh_tour = get_lkh_results_dists(d, not params["asymmetric"])
            heur_length = compute_tour_length(instance[2], lkh_tour)
            if best_pi is None or not set(best_pi) == set(range(data.num_nodes)):
                print(f"RL solution invalid {instance[0]}")
                RL_length = float("inf")
            else:
                RL_length = compute_tour_length(instance[2], best_pi)

            assert set(lkh_tour) == set(range(data.num_nodes)), "LKH solution invalid"

        elif params["problem"] == "CVRP":
            if best_pi is not None:
                assert is_valid_cvrp(instance[3], instance[4], best_pi)
                RL_length = compute_tour_length(instance[2], best_pi)
            else:
                RL_length = float("inf")

            path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, "libresults.json")
            results = load_or_create_json(path)

            if (
                "CVRP" in results
                and instance[0] in results["CVRP"]
                and "heuristic_length" in results["CVRP"][instance[0]]
            ):
                heur_length = results["CVRP"][instance[0]]["heuristic_length"]
            else:
                hgs_tour = get_hgs_results_dists_cvrp(
                    d, instance[3], instance[4], not params["asymmetric"]
                )
                assert is_valid_cvrp(instance[3], instance[4], hgs_tour)
                heur_length = compute_tour_length(instance[2], hgs_tour)

        elif params["problem"] == "OP":
            if best_pi is not None:
                RL_rewards = compute_collected_prize(best_pi, instance[3])
                assert is_valid_op(best_pi, instance[2], instance[4])
            else:
                RL_rewards = float("-inf")
            heur_rewards = best_known_length
            heur_length = best_known_length

        if params["problem"] in ["TSP", "CVRP"]:
            rl_gap = round(((RL_length / best_known_length) - 1) * 100, 2)
            heur_gap = round(((heur_length / best_known_length) - 1) * 100, 2)
            rl_gap_wrt_heur = round(((RL_length / heur_length) - 1) * 100, 2)

            heuristic = None
            if params["problem"] == "TSP":
                heuristic = "LKH"
            elif params["problem"] == "CVRP":
                heuristic = "HGS"
            elif params["problem"] == "OP":
                heuristic = "EA4OP"

            print(
                f"Evaluating instance {instance[0]} which has {data.num_nodes} cities."
            )
            print(f"Length best known: {round(best_known_length, 2)}")
            print(f"Length RL: {round(RL_length, 2)}")
            print(f"Length {heuristic}: {round(heur_length, 2)}")
            print(f"GAP {heuristic} wrt best: {heur_gap}%")
            print(f"GAP RL wrt best: {rl_gap}%")
            print(f"GAP RL wrt {heuristic}: {rl_gap_wrt_heur}%")
            print("-------------------------------------------------------")
        else:
            print(
                f"Evaluating instance {instance[0]} which has {data.num_nodes} cities."
            )
            rl_gap = round(((heur_rewards / RL_rewards) - 1) * 100, 2)
            if RL_rewards == float("-inf"):
                rl_gap = float("inf")
            rl_gap_wrt_heur = (
                rl_gap  # best is EA4OP anyway so no need to call it outselves
            )
            heur_gap = 0
            print(f"Prize best known: {round(best_known_length, 2)}")
            print(f"Prize RL: {round(RL_rewards, 2)}")
            print(f"GAP RL wrt best: {rl_gap}%")
            print("-------------------------------------------------------")
        save_result(
            model_name,
            instance[0],
            rl_gap,
            rl_gap_wrt_heur,
            heur_gap,
            heur_length,
            params,
        )


if __name__ == "__main__":
    eval()
