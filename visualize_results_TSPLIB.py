import gzip
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from great.utils.constants import BENCHMARKING_RL_CONFIGS_PATH, DATA_PATH, FIGURE_PATH

### this file is for visualizing the results that our trained model achieved on the difference real world benchmark datasets
### TSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
### ATSPLIB http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
### CVRPLib http://vrp.atd-lab.inf.puc-rio.br/index.php/en/ (instances with more than 200 nodes as well as XML100 are not considered)
### OPLib https://github.com/bcamath-ds/OPLib/tree/master


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


def create_statistics(results, filter_threshold=float("inf")):
    statistics = {}
    for model in results:
        gaps = results[model]
        gaps = [x for x in gaps if not math.isinf(x)]
        gaps = [x for x in gaps if x <= filter_threshold]
        arr = np.array(gaps)

        mean = np.mean(arr)
        median = np.median(arr)
        q10 = np.percentile(arr, 10)
        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        q90 = np.percentile(arr, 90)
        num_values = len(gaps)

        statistics[model] = {
            "mean": mean,
            "median": median,
            "q10": q10,
            "q25": q25,
            "q75": q75,
            "q90": q90,
            "num_values": num_values,
        }
    return statistics


def visualize_statistics(statistics):
    names = []
    for model in statistics:
        names.append(model)
    names = sorted(names)
    names = sorted(names, key=lambda s: "MIX" in s)
    names = sorted(names, key=lambda s: "matnet" in s)
    names = sorted(names, key=lambda s: "pointerformer" in s)
    problem = names[0]
    problem = problem.split("_")[1]

    means = [statistics[model]["mean"] for model in names]
    medians = [statistics[model]["median"] for model in names]
    q10 = [statistics[model]["q10"] for model in names]
    q25 = [statistics[model]["q25"] for model in names]
    q75 = [statistics[model]["q75"] for model in names]
    q90 = [statistics[model]["q90"] for model in names]
    num_values = [statistics[model]["num_values"] for model in names]
    assert (
        len(set(num_values)) == 1
    ), "All models should be evaluated on the same amount of test instances!"
    print(f"Number of evaluated instances: {num_values[0]}")

    x = list(range(len(names)))

    # Plot each list
    plt.plot(x, means, label="Mean")
    plt.plot(x, medians, label="Median")
    plt.plot(x, q10, label="Q10")
    plt.plot(x, q25, label="Q25")
    plt.plot(x, q75, label="Q75")
    plt.plot(x, q90, label="Q90")

    # Clean x-axis labels
    cleaned_names = [s.replace("_" + problem, "") for s in names]
    plt.xticks(ticks=x, labels=cleaned_names, rotation=45)

    # Logarithmic y-axis
    plt.yscale("log")
    plt.ylim(0.1, 200)  # Set limits to avoid log(0)

    if any("TMAT" in s or "XASY" in s for s in names):
        problem = "A" + problem

    # Add labels and legend
    plt.xlabel("Model")
    plt.ylabel("Gap")
    plt.title(problem)
    plt.legend()

    plt.tight_layout()  # Prevent label cutoff
    plt.savefig(os.path.join(FIGURE_PATH, problem + ".png"))
    plt.clf()

    def second_smallest(vals):
        unique_sorted = sorted(set(vals))
        return unique_sorted[1] if len(unique_sorted) > 1 else None

    # Round and prepare all stat columns first
    mean_vals = [round(m, 2) for m in means]
    q10_vals = [round(m, 2) for m in q10]
    q25_vals = [round(m, 2) for m in q25]
    median_vals = [round(m, 2) for m in medians]
    q75_vals = [round(m, 2) for m in q75]
    q90_vals = [round(m, 2) for m in q90]

    # Min and 2nd min per column
    min_mean, second_mean = min(mean_vals), second_smallest(mean_vals)
    min_q10, second_q10 = min(q10_vals), second_smallest(q10_vals)
    min_q25, second_q25 = min(q25_vals), second_smallest(q25_vals)
    min_median, second_median = min(median_vals), second_smallest(median_vals)
    min_q75, second_q75 = min(q75_vals), second_smallest(q75_vals)
    min_q90, second_q90 = min(q90_vals), second_smallest(q90_vals)

    cleaned_names_for_table = []
    distributions = []
    for name in cleaned_names:
        if "matnet" in name.lower():
            cleaned_names_for_table.append("& MatNet x9")
        if "_NB" in name:
            cleaned_names_for_table.append("& GREAT NB x9")
        if "_NF" in name:
            cleaned_names_for_table.append("& GREAT NF x9")
        if "pointerformer" in name:
            cleaned_names_for_table.append("& Pointerformer x8")

        distributions.append(name.split("_", 1)[0])

    # Build formatted rows
    rows = []
    for i in range(len(cleaned_names)):
        row = [cleaned_names_for_table[i]] + [distributions[i]]

        stats = [
            (mean_vals[i], min_mean, second_mean),
            (q10_vals[i], min_q10, second_q10),
            (q25_vals[i], min_q25, second_q25),
            (median_vals[i], min_median, second_median),
            (q75_vals[i], min_q75, second_q75),
            (q90_vals[i], min_q90, second_q90),
        ]

        for val, col_min, col_second in stats:
            if val == col_min:
                formatted = f"\\textbf{{{val}\\%}}"
            elif val == col_second:
                formatted = f"\\underline{{{val}\\%}}"
            else:
                formatted = f"{val}\\%"
            row.append(formatted)

        rows.append(row)
    print(problem)

    # Print header
    print("Model & Dist. & Mean & Q10 & Q25 & Median & Q75 & Q90 \\\\")

    # Print rows
    for row in rows:
        print(" & ".join(row) + " \\\\")


def get_instance_sizes():
    tsp_instances = [
        os.path.abspath(os.path.join(os.path.join(DATA_PATH, "ALL_tsp"), f))
        for f in os.listdir(os.path.join(DATA_PATH, "ALL_tsp"))
        if os.path.isfile(os.path.join(os.path.join(DATA_PATH, "ALL_tsp"), f))
    ]
    tsp_instances = [f for f in tsp_instances if ".tsp.gz" in f]

    atsp_instances = [
        os.path.abspath(os.path.join(os.path.join(DATA_PATH, "ALL_atsp"), f))
        for f in os.listdir(os.path.join(DATA_PATH, "ALL_atsp"))
        if os.path.isfile(os.path.join(os.path.join(DATA_PATH, "ALL_atsp"), f))
    ]
    atsp_instances = [f for f in atsp_instances if ".atsp.gz" in f]

    vrp_instances = []
    for root, dirs, files in os.walk(os.path.join(DATA_PATH, "ALL_VRP")):
        for file in files:
            vrp_instances.append(os.path.join(root, file))
    vrp_instances = [f for f in vrp_instances if ".vrp" in f]

    op_instances = []
    for root, dirs, files in os.walk(os.path.join(DATA_PATH, "ALL_OP")):
        for file in files:
            op_instances.append(os.path.join(root, file))
    op_instances = [f for f in op_instances if ".oplib" in f]

    instances = tsp_instances + atsp_instances + vrp_instances
    sizes = dict()
    for file_path in instances:
        if ".gz" in file_path:
            with gzip.open(file_path, "rt") as f:  # 'rt' = read text mode
                name = None
                for line in f:
                    if line[:4] == "NAME":
                        name = line.split()[-1]
                    if "DIMENSION" in line:
                        dimension = line.split()[-1]
                        dimension = int(dimension)
                        assert name is not None
                        if name in sizes:
                            print(f"double instance occurence {name}")
                            assert False
                        sizes[name] = dimension
                        break
        else:
            with open(file_path, "r") as f:
                name = None
                for line in f:
                    if line[:4] == "NAME":
                        name = line.split()[-1]
                    if "DIMENSION" in line:
                        dimension = line.split()[-1]
                        dimension = int(dimension)
                        assert name is not None
                        if name in sizes:
                            print(f"double instance occurence {name}")
                            assert False
                        sizes[name] = dimension
                        break

    for file_path in op_instances:
        with open(file_path, "r") as f:
            name = os.path.basename(file_path)[:-6]
            for line in f:
                if "DIMENSION" in line:
                    dimension = line.split()[-1]
                    dimension = int(dimension)
                    if name in sizes:
                        print(f"double instance occurence {name}")
                        assert False
                    sizes[name] = dimension
                    break
    return sizes


def reduce_results(total_results, instance_sizes, sizes_to_keep):
    problems = ["TSP", "CVRP", "OP"]

    shrunken_results = dict()
    for p in problems:
        shrunken_results[p] = dict()
        for instance in total_results[p]:
            if (
                instance_sizes[instance] >= sizes_to_keep[0]
                and instance_sizes[instance] <= sizes_to_keep[1]
            ):
                shrunken_results[p][instance] = total_results[p][instance]
    return shrunken_results


if __name__ == "__main__":
    path = os.path.join(BENCHMARKING_RL_CONFIGS_PATH, "libresults.json")
    results = load_or_create_json(path)

    sizes = get_instance_sizes()

    considered_sizes = (101, 200)

    if considered_sizes[1] > 100:
        exclude_matnet = True
    else:
        exclude_matnet = False

    results = reduce_results(results, sizes, considered_sizes)

    ### Idea: create boxplots for each Lib - Model combination
    models_to_consider_TSP = {
        "EUC_TSP_NB",
        "EUC_TSP_NF",
        "MIX_TSP_NB",
        "MIX_TSP_NF",
        "EUC_TSP_matnet",
        "EUC_TSP_pointerformer",  # TSPLIB
    }
    models_to_consider_ATSP = {
        "TMAT_TSP_NB",
        "TMAT_TSP_NF",
        "XASY_TSP_NB",
        "XASY_TSP_NF",
        "MIX_TSP_NB",
        "MIX_TSP_NF",
        "XASY_TSP_matnet",
        "TMAT_TSP_matnet",  # ATSPLIB
    }
    models_to_consider_CVRP = {
        "EUC_CVRP_NB",
        "EUC_CVRP_NF",
        "MIX_CVRP_NB",
        "MIX_CVRP_NF",
        "EUC_CVRP_matnet",
        "EUC_CVRP_pointerformer",  # CVRPLIB
    }
    models_to_consider_OP = {
        "EUC_OP_NB",
        "EUC_OP_NF",
        "MIX_OP_NB",
        "MIX_OP_NF",
        "EUC_OP_matnet",
        "EUC_OP_pointerformer",  # OPLIB
    }

    if exclude_matnet:
        models_to_consider_TSP = {
            s for s in models_to_consider_TSP if "matnet" not in s
        }
        models_to_consider_ATSP = {
            s for s in models_to_consider_ATSP if "matnet" not in s
        }
        models_to_consider_CVRP = {
            s for s in models_to_consider_CVRP if "matnet" not in s
        }
        models_to_consider_OP = {s for s in models_to_consider_OP if "matnet" not in s}

    TSP_results = {}
    for model in models_to_consider_TSP:
        if model not in TSP_results:
            TSP_results[model] = list()
        for instance in results["TSP"]:
            if "EUC_TSP_NB" in results["TSP"][instance]:  ### it is TSP and not ATSP
                if model in results["TSP"][instance]:
                    TSP_results[model].append(
                        results["TSP"][instance][model]["rl_gap_wrt_heur"]
                    )

    ATSP_results = {}
    for model in models_to_consider_ATSP:
        if model not in ATSP_results:
            ATSP_results[model] = list()
        for instance in results["TSP"]:
            if "TMAT_TSP_NB" in results["TSP"][instance]:  ### it is ATSP and not TSP
                if model in results["TSP"][instance]:
                    ATSP_results[model].append(
                        results["TSP"][instance][model]["rl_gap_wrt_heur"]
                    )

    CVRP_results = {}
    for model in models_to_consider_CVRP:
        if model not in CVRP_results:
            CVRP_results[model] = list()
        for instance in results["CVRP"]:
            if model in results["CVRP"][instance]:
                CVRP_results[model].append(
                    results["CVRP"][instance][model]["rl_gap_wrt_heur"]
                )

    OP_results = {}
    for model in models_to_consider_OP:
        if model not in OP_results:
            OP_results[model] = list()
        for instance in results["OP"]:
            if model in results["OP"][instance]:
                OP_results[model].append(
                    results["OP"][instance][model]["rl_gap_wrt_heur"]
                )

    print(f"Results for sizes instances of size {considered_sizes}")
    visualize_statistics(create_statistics(CVRP_results))
    visualize_statistics(create_statistics(TSP_results))
    visualize_statistics(create_statistics(ATSP_results))
    visualize_statistics(create_statistics(OP_results))
