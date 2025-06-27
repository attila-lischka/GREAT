import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from great.utils.constants import BENCHMARKING_RL_CONFIGS_PATH
from great.utils.utils import (
    get_all_trained_models_configs,
    get_matching_config,
    get_model,
    get_model_file,
    get_traindataset,
    save_model,
    set_hyperparams,
    set_seeds,
)


def train_loop(
    model,
    loader,
    device,
    optimizer,
    params,
):
    for data in tqdm(loader):
        data.to(device)
        optimizer.zero_grad()
        if params["task"] == "RL":
            outputs = model(data, augmentation_factor=params["augmentation_factor"])
            loss = outputs
        else:
            outputs = model(data)
            loss = model.apply_criterion(
                outputs,
                data,
                criterion=params["criterion"],
            )
        loss.backward()
        optimizer.step()


@hydra.main(
    version_base=None,
    config_path=BENCHMARKING_RL_CONFIGS_PATH,
    config_name="TMAT_TSP_NB",
)
def run(cfg: DictConfig):
    """
    This script is to fine tune an existing model in the RL task to bigger instances
    """
    ### setting seeds
    set_seeds(1234)

    increment_factor = (
        1.1  # factor problem instance sizes are increased by each iteration
    )
    target_size = 500
    start_size = 100

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
    params["timestamp"] = model_time_stamp  # overwrite timestamp value
    print(f"Retraining trained model with timestamp: {model_time_stamp}")

    # get the file where the model with the matching ID is stored and load it
    model_file = get_model_file(model_time_stamp)
    model.load_state_dict(torch.load(model_file, map_location=params["device"]))

    # init the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=0.000001
    )

    fine_tuning_sizes = [200, 500]
    size = start_size * increment_factor
    while size < target_size:
        fine_tuning_sizes.append(int(size))
        size = size * increment_factor

    fine_tuning_sizes.sort()

    for size in fine_tuning_sizes:
        print(f"Retraining for size: {size}")
        params["instance_size"] = size
        params["dataset_size"] = 2000

        train_dataset = get_traindataset(
            params, dataset_seed=0, save_processed_data=True, save_raw_data=True
        )

        if size < 250:
            batch_size = 16
        elif size < 350:
            batch_size = 8
        else:
            batch_size = 4

        if size in [
            200,
            500,
        ]:  # the instance sizes we want to specialize on are trained more extensively
            epochs = 5
        else:
            epochs = 1

        for epoch in range(epochs):
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            model.train(True)
            train_loop(
                model,
                train_loader,
                params["device"],
                optimizer,
                params,
            )

        if size in [200, 500, 1000]:
            save_model(model, f"CL_{size}", params, [], False)


if __name__ == "__main__":
    run()
