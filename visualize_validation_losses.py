import os
import re

import matplotlib.pyplot as plt

from great.utils.constants import FINAL_MODEL_PATH


def get_val_losses_from_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        line = lines[-1]
        line = line.split()[1:]

        line = [
            re.sub(r"[^\d.]", "", s) for s in line
        ]  # remove everything that is not a number
        line = [float(x) for x in line]

    return line


def visualize_loss(file_name, loss_list):
    plt.plot(loss_list)
    plt.title("Validation Loss Development")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    base = os.path.splitext(file_name)[0] + ".png"

    plt.savefig(base)  # Save the figure
    plt.close()


if __name__ == "__main__":
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

    losses = list()
    for f in files:
        losses.append((f, get_val_losses_from_file(f)))

    for f, line in losses:
        visualize_loss(f, line)
