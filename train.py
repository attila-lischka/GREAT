import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from great.utils.constants import TRAIN_CONFIG_PATH
from great.utils.utils import (
    delete_old_model,
    get_model,
    get_traindataset,
    get_valdataset,
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


def val_loop(model, val_loader, device, params):
    total_vloss = 0.0
    with torch.no_grad():
        for vdata in val_loader:
            vdata.to(device)

            if params["task"] == "RL":
                reward_ntraj, _ = model.get_tour(
                    vdata, augmentation_factor=params["augmentation_factor"]
                )
                vloss = reward_ntraj  # we want to track how long the tours in the validation data were
                num_instances = vdata.num_nodes / params["instance_size"]

                total_vloss += torch.tensor((vloss * num_instances))

            else:
                outputs = model(vdata)
                vloss = model.apply_criterion(
                    outputs,
                    vdata,
                    criterion=params["criterion"],
                )
                num_instances = vdata.num_nodes / params["instance_size"]
                total_vloss += vloss / num_instances
        return total_vloss.item()


@hydra.main(
    version_base=None, config_path=TRAIN_CONFIG_PATH, config_name="EUC_CVRP_matnet"
)
def run(cfg: DictConfig):
    """
    This is to train a new model which will be saved later
    """
    ### setting seeds
    set_seeds(1234)

    ### set hyperparameters
    params = set_hyperparams(cfg)

    ### init model
    model = get_model(params)
    model.to(params["device"])  # move to device

    # print model statistics
    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}"
        )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # init the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=0.000001
    )

    # get a validation dataset
    val_dataset = get_valdataset(params)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # start training
    print("Start Model Training")
    dataset_seed = 0  ### tracker which dataset to pick
    best_epoch = (
        0  # tracker in which epoch the model performed best on the validation data
    )
    if params["problem"] == "OP":
        best_val_loss = float("-inf")
    else:
        best_val_loss = float("inf")  # loss of the best epoch
    val_losses = []  # list to save all validation losses during training (can be interesting to plot)
    assert (
        params["epochs"] % params["num_datasets"] == 0
    ), "number epochs not divisible by number datasets"
    factor_dataset = params["epochs"] // params["num_datasets"]
    factor_dataset_counter = 0

    for epoch in range(params["epochs"]):
        if (
            factor_dataset_counter % factor_dataset == 0
        ):  # every factor_ds datasets, change the dataset (we could also change every epoch but this way we don't need to generate a new one that often and therefore training is faster)
            train_dataset = get_traindataset(params, dataset_seed=dataset_seed)
            dataset_seed += 1

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"])

        model.train(True)
        train_loop(
            model,
            train_loader,
            params["device"],
            optimizer,
            params,
        )

        model.eval()

        val_loss = val_loop(
            model,
            val_loader,
            params["device"],
            params,
        )

        val_losses.append(val_loss)

        # if there is a new best epoch
        if params["problem"] == "OP":
            if val_loss > best_val_loss:  # OP we want to maximize collected prizes
                save_model(model, epoch, params, val_losses)
                delete_old_model(old_best_epoch=best_epoch, params=params)
                best_val_loss = val_loss
                best_epoch = epoch
        else:
            if val_loss < best_val_loss:
                save_model(model, epoch, params, val_losses)
                delete_old_model(old_best_epoch=best_epoch, params=params)
                best_val_loss = val_loss
                best_epoch = epoch

        # increment dataset counter
        factor_dataset_counter += 1

    # save final model after training (might not be the best one!)
    save_model(model, "final", params, val_losses)


if __name__ == "__main__":
    run()
