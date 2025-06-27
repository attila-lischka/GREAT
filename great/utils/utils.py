import os
import random
from datetime import datetime

import numpy as np
import torch

from great.utils.constants import FINAL_MODEL_PATH, MODEL_PATH

from ..dataset.dataset import GREATRLDataset, GREATTSPDataset
from ..models.cvrp_model import GREATRL_CVRP
from ..models.great import GREAT
from ..models.op_model import GREATRL_OP
from ..models.tsp_model import GREATRL_TSP


def set_seeds(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def set_hyperparams(cfg):
    """
    Create a dict of hyperparameters given the config
    """

    params = dict()

    ### copy config
    for k in cfg:
        params[k] = cfg[k]

    ### set a training device
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### create a unique ID for the task
    params["timestamp"] = (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid())
    )

    ### set variables depending on task
    if params["task"] == "tour":
        if params["scale_weights"]:  # use scaled weights for the loss
            positives = (
                params["instance_size"]
                * 2
                / (
                    params["instance_size"] * params["instance_size"]
                    - params["instance_size"]
                )
            )  # compute fraction of edges in a complete graph that are part of the TSP solution
            class0_weight = 1.0 / (1 - positives)  # Weight for class 0
            class1_weight = 1.0 / positives  # Weight for class 1
            weights = torch.tensor([class0_weight, class1_weight]).to(params["device"])
        else:
            weights = torch.tensor([1.0, 1.0]).to(params["device"])
        params["weights"] = weights
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    elif params["task"] == "cost":
        criterion = torch.nn.MSELoss()
    elif params["task"] == "RL":
        criterion = None  ### RL has its own, custom loss formulation
    else:
        raise NotImplementedError("Unknown task: " + str(params["task"]))

    if (
        "matnet" in params and "pointerformer" in params
    ):  # must not be true at the same time
        assert not (
            params["matnet"] and params["pointerformer"]
        ), "Pointerformer and Matnet cannot be used at the same time"

    if "pointerformer" in params:
        assert (
            params["data_distribution"] == "EUC"
        ), "Pointerformer is only compatible with EUC distribution"
    else:
        params["pointerformer"] = False
    params["criterion"] = criterion

    # is the data asymmetric?
    if (
        "ASY" in params["data_distribution"]
        or params["data_distribution"] == "TMAT"
        or params["data_distribution"] == "MIX"
        or params["problem"]
        in [
            "OP",
            "CVRP",
        ]  # since customer i and j have different demands, edge e_ij is different from e_ji
    ):
        asymmetric = True
    else:
        asymmetric = False
    params["asymmetric"] = asymmetric

    return params


def get_model(params):
    """
    Get the correct model depending on the learning task
    """

    if params["task"] == "tour":
        model = GREAT(
            initial_dim=5,
            hidden_dim=params["hidden_dim"],
            heads=params["heads"],
            num_layers=params["num_layers"],
            num_nodes=params["instance_size"],
            num_classes=2,
            nodeless=params["nodeless"],
        )
    elif params["task"] == "cost":
        model = GREAT(
            initial_dim=5,
            hidden_dim=params["hidden_dim"],
            heads=params["heads"],
            num_layers=params["num_layers"],
            num_nodes=params["instance_size"],
            regression=True,
            nodeless=params["nodeless"],
            instance_repr=True,
        )
    elif params["task"] == "RL":
        if params["problem"] == "TSP":
            model = GREATRL_TSP(
                initial_dim=5,
                hidden_dim=params["hidden_dim"],
                heads=params["heads"],
                num_layers=params["num_layers"],
                num_nodes=params["instance_size"],
                group_size=params["instance_size"],
                final_node_layer=params["final_node_layer"],
                nodeless=params["nodeless"],
                asymmetric=params["asymmetric"],
                matnet=params["matnet"],
                pointerformer=params["pointerformer"],
            )
        elif params["problem"] == "CVRP":
            model = GREATRL_CVRP(
                initial_dim=6,  # one additional dim compared to TSP which encodes the demand of each node
                hidden_dim=params["hidden_dim"],
                heads=params["heads"],
                num_layers=params["num_layers"],
                num_nodes=params["instance_size"],
                group_size=params["instance_size"],
                final_node_layer=params["final_node_layer"],
                nodeless=params["nodeless"],
                asymmetric=params["asymmetric"],
                matnet=params["matnet"],
                pointerformer=params["pointerformer"],
            )
        elif params["problem"] == "OP":
            model = GREATRL_OP(
                initial_dim=8,  # two additional dim compared to TSP which encode the prize for each node and its distance to return to the depot
                hidden_dim=params["hidden_dim"],
                heads=params["heads"],
                num_layers=params["num_layers"],
                num_nodes=params["instance_size"],
                group_size=params["instance_size"],
                final_node_layer=params["final_node_layer"],
                nodeless=params["nodeless"],
                asymmetric=params["asymmetric"],
                matnet=params["matnet"],
                xasy=params["data_distribution"] in ["XASY", "MIX"],
                pointerformer=params["pointerformer"],
            )
        else:
            raise NotImplementedError("Unknown problem " + str(params["problem"]))
    else:
        raise NotImplementedError("Unknown task: " + str(params["task"]))

    return model


def get_valdataset(params):
    """
    Get a validation dataset for the training depending on the task
    """

    if params["task"] in ["RL"]:
        val_dataset = GREATRLDataset(
            data_dist=params["data_distribution"],
            dataset_size=1000,
            problem=params["problem"],
            instance_size=params["instance_size"],
            seed=9999,
        )
    elif params["task"] in ["tour", "cost"]:
        assert (
            params["problem"] == "TSP"
        ), f"Task {params['task']} only supports TSP and not {params['problem']}"
        assert (
            params["data_distribution"] == "EUC"
        ), f"Task {params['task']} only supports EUC distances for now and not {params['data_distribution']}"
        val_dataset = GREATTSPDataset(
            dataset_size=1000,
            tsp_size=params["instance_size"],
            seed=9999,
            save_raw_data=True,
        )
    else:
        raise NotImplementedError("Unknown task: " + str(params["task"]))
    return val_dataset


def get_traindataset(
    params, dataset_seed, save_processed_data=False, save_raw_data=False
):
    """
    Get a training dataset depending on the task
    """
    if params["task"] in ["RL"]:
        train_dataset = GREATRLDataset(
            data_dist=params["data_distribution"],
            dataset_size=params["dataset_size"],
            problem=params["problem"],
            instance_size=params["instance_size"],
            seed=dataset_seed,
            save_processed_data=save_processed_data,
            save_raw_data=save_raw_data,
        )
    elif params["task"] in ["tour", "cost"]:
        assert (
            params["problem"] == "TSP"
        ), f"Task {params['task']} only supports TSP and not {params['problem']}"
        assert (
            params["data_distribution"] == "EUC"
        ), f"Task {params['task']} only supports EUC distances for now and not {params['data_distribution']}"

        train_dataset = GREATTSPDataset(
            dataset_size=params["dataset_size"],
            tsp_size=params["instance_size"],
            seed=dataset_seed,
        )
    else:
        raise NotImplementedError("Unknown task: " + str(params["task"]))

    return train_dataset


def get_model_folder():
    # get folder where to save the model
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    return MODEL_PATH


def write_model_infos(params, root, loss):
    # save some additional information about the model training in a txt file
    file_name = "/training_" + params["timestamp"] + ".txt"
    file_path = root + file_name
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as text_file:
        for k, v in params.items():
            text_file.write(str(k) + " >>> " + str(v) + "\n\n")
        text_file.write("\n")
        text_file.write("Validation loss:")
        text_file.write(str(loss))


def save_model(model, epoch, params, val_losses, save_model_infos=True):
    # get folder where to save the model
    root = get_model_folder()

    # save the model
    model_path = "/model_{}_{}.ckpt".format(params["timestamp"], epoch)
    torch.save(model.state_dict(), root + model_path)
    if save_model_infos:
        write_model_infos(params, root + "/train_infos", val_losses)


def delete_old_model(old_best_epoch, params):
    # get folder where to save the model
    root = get_model_folder()

    if os.path.exists(
        root + "/model_{}_{}.ckpt".format(params["timestamp"], old_best_epoch)
    ):
        os.remove(
            root + "/model_{}_{}.ckpt".format(params["timestamp"], old_best_epoch)
        )


def cast_to_correct_type(string):
    # Try to cast to boolean
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False

    # Try to cast to integer
    try:
        return int(string)
    except ValueError:
        pass

    # Try to cast to float
    try:
        return float(string)
    except ValueError:
        pass

    # If all casts fail, return the string itself
    return string


def get_config_from_file(filename):
    params = dict()
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if len(line) < 3:
                continue
            if line[0] == "Validation":
                continue
            if line[0] == "weights":
                temp = line[2:]
                temp = "".join(temp)
                temp = temp[8:]
                temp = temp.split("]")
                temp = temp[0]
                temp = temp.split(",")
                temp = [float(x) for x in temp]
                temp = torch.tensor(temp)
                params[line[0]] = temp
            else:
                params[line[0]] = line[2]
        for k in params:
            if isinstance(params[k], str) and k != "timestamp":
                params[k] = cast_to_correct_type(params[k])
        return params


def get_all_trained_models_configs():
    files = []
    # Get direct subfolders
    subfolders = [
        os.path.join(FINAL_MODEL_PATH, d)
        for d in os.listdir(FINAL_MODEL_PATH)
        if os.path.isdir(os.path.join(FINAL_MODEL_PATH, d))
    ]

    # Loop through each subfolder to get files
    for subfolder in subfolders:
        entries = os.listdir(subfolder)
        for entry in entries:
            entry_path = os.path.join(subfolder, entry)
            if os.path.isfile(entry_path):
                files.append(entry_path)

    if ".DS_Store" in files:
        files.remove(".DS_Store")
    files = [f for f in files if ".txt" in f]
    files = [f for f in files if "toursx" not in f]  # remove the solution files
    files = [f for f in files if "HGS.txt" not in f]  # remove the solution files
    files = [f for f in files if "solution.txt" not in f]  # remove the solution files
    files = [f for f in files if "test_concorde" not in f]  # remove the solution files
    files = [f for f in files if "_heuristic.txt" not in f]  # remove the solution files

    configs = list()
    for f in files:
        conf = get_config_from_file(f)
        if "classification" not in conf:
            conf["classification"] = "edge_wise_class"
        if "nodeless" not in conf:
            conf["nodeless"] = None
        if "matnet" not in conf:
            conf["matnet"] = False
        if "pointerformer" not in conf:
            conf["pointerformer"] = False
        configs.append(conf)
    return configs


def get_matching_config(configs, params):
    temp_params = dict()
    for k in params:
        if (
            k != "timestamp" and k != "device" and k != "criterion"
        ):  # these shall be ignored for matching
            temp_params[k] = params[k]

    matching_configs = list()
    for config in configs:
        match = True
        for k in temp_params:
            if k == "weights":
                if not torch.allclose(temp_params[k], config[k], atol=1e-3):
                    match = False
                    break
            elif config[k] != temp_params[k]:
                match = False
                break
        if match:
            matching_configs.append(config)

    assert len(matching_configs) == 1, (
        "Found " + str(len(matching_configs)) + " many matching configs!"
    )

    return matching_configs[0]


def get_model_file(timestamp, CL=None):
    files = []
    # Get direct subfolders
    subfolders = [
        os.path.join(FINAL_MODEL_PATH, d)
        for d in os.listdir(FINAL_MODEL_PATH)
        if os.path.isdir(os.path.join(FINAL_MODEL_PATH, d))
    ]

    # Loop through each subfolder to get files
    for subfolder in subfolders:
        entries = os.listdir(subfolder)
        for entry in entries:
            entry_path = os.path.join(subfolder, entry)
            if os.path.isfile(entry_path):
                files.append(entry_path)

    files = [f for f in files if ".ckpt" in f]
    files = [f for f in files if timestamp in f]
    if CL is None:
        files = [f for f in files if "_CL_" not in f]
    else:
        files = [f for f in files if f"_CL_{CL}" in f]

    assert len(files) == 1, (
        "Found "
        + str(len(files))
        + " matching models!"
        + str(files)
        + " with timestamp "
        + timestamp
    )

    return files[0]
