import json
import os
import subprocess
from datetime import datetime

import numpy as np

from great.utils.constants import LKH_SOLVER_PATH

"""
A file containing wrapper functions to call the LKH algorithm to solve TSP (EUC and ASY)
Further, the file contains wrapper functions to call the HGS algorithm to solve CVRP (EUC and ASY)
"""


def get_lkh_results(nodes):
    def getDistance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    pid = os.getpid()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    pid = str(pid) + "_" + current_time
    par_file_name = "problem_tsp_temp" + pid + ".par"
    candidate_file_name = "cand_tsp_temp" + pid + ".txt"
    instance_file_name = "instance_tsp_temp" + pid + ".txt"
    tour_file_name = "tour_tsp_temp" + pid + ".txt"

    path_absolute = os.path.dirname(os.path.abspath(__file__))

    tourlength = len(nodes)

    with open(path_absolute + "/" + par_file_name, "w") as file:
        file.write("PROBLEM_FILE = " + path_absolute + "/" + instance_file_name + "\n")
        file.write("TOUR_FILE = " + path_absolute + "/" + tour_file_name + "\n")
        file.write("CANDIDATE_SET_TYPE = ALPHA \n")

    with open(path_absolute + "/" + instance_file_name, "w") as file:
        file.write("NAME : " + instance_file_name[: len(instance_file_name) - 4] + "\n")
        file.write(
            "COMMENT : random, uniform data instance located in unit square, scaled \n"
        )
        file.write("TYPE : TSP\n")
        file.write("DIMENSION : " + str(tourlength) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for i, node in enumerate(nodes):
            file.write(str(i + 1))
            file.write(" ")
            file.write(str(node[0] * 1000))
            file.write(" ")
            file.write(str(node[1] * 1000))
            file.write("\n")

        file.write("EOF")

    subprocess.call(
        [LKH_SOLVER_PATH + "/LKH", path_absolute + "/" + par_file_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    with open(path_absolute + "/" + tour_file_name) as f:
        lines = f.readlines()

    length = lines[1]
    length = int(length.split()[-1]) / 1000
    lines = lines[6 : len(lines) - 2]
    tour = [int(line.split()[0]) - 1 for line in lines]

    if os.path.exists(path_absolute + "/" + candidate_file_name):
        os.remove(path_absolute + "/" + candidate_file_name)

    if os.path.exists(path_absolute + "/" + par_file_name):
        os.remove(path_absolute + "/" + par_file_name)

    if os.path.exists(path_absolute + "/" + instance_file_name):
        os.remove(path_absolute + "/" + instance_file_name)
    if os.path.exists(path_absolute + "/" + tour_file_name):
        os.remove(path_absolute + "/" + tour_file_name)
    return tour


def get_lkh_results_dists_cvrp(
    dists, demands, capacity, symmetric=False, TIME_LIMIT=None
):
    pid = os.getpid()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    pid = str(pid) + "_" + current_time
    par_file_name = "problem_tsp_temp" + pid + ".par"
    candidate_file_name = "cand_tsp_temp" + pid + ".txt"
    instance_file_name = "instance_cvrp_temp" + pid + ".vrp"
    tour_file_name = "tour_tsp_temp" + pid + ".tour"

    path_absolute = os.path.dirname(os.path.abspath(__file__))

    tourlength = len(dists)

    with open(path_absolute + "/" + par_file_name, "w") as file:
        file.write("PROBLEM_FILE = " + path_absolute + "/" + instance_file_name + "\n")
        file.write("TOUR_FILE = " + path_absolute + "/" + tour_file_name + "\n")
        # file.write("CANDIDATE_SET_TYPE = ALPHA \n")
        if TIME_LIMIT is not None:
            file.write("TIME_LIMIT = " + str(TIME_LIMIT) + "\n")

    with open(path_absolute + "/" + instance_file_name, "w") as file:
        file.write("NAME : " + instance_file_name[: len(instance_file_name) - 4] + "\n")
        file.write("COMMENT : random data, scaled \n")
        if symmetric:
            file.write("TYPE : CVRP\n")
        else:
            file.write("TYPE : ACVRP\n")
        file.write("DIMENSION : " + str(tourlength) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        file.write("CAPACITY : " + str(capacity) + "\n")
        if not symmetric:
            file.write("VEHICLES : " + str(tourlength) + "\n")
        file.write("EDGE_WEIGHT_SECTION\n")
        for i, dist in enumerate(dists):
            for d in dist:
                file.write(str(int(d * 1000)))
                file.write(" ")
            file.write("\n")
        file.write("DEMAND_SECTION\n")
        for i, dem in enumerate(demands):
            file.write(str(i + 1) + "  " + str(dem) + "\n")
        file.write("DEPOT_SECTION\n")
        file.write("1\n")
        file.write("EOF")

    subprocess.call(
        [LKH_SOLVER_PATH + "/LKH", path_absolute + "/" + par_file_name],
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.STDOUT,
    )

    with open(path_absolute + "/" + tour_file_name) as f:
        lines = f.readlines()

    length = lines[1]
    length = int(length.split()[-1]) / 1000
    lines = lines[6 : len(lines) - 2]
    tour = [int(line.split()[0]) - 1 for line in lines]
    tour = [x if x < tourlength else 0 for x in tour]
    if tour[0] != 0:
        tour.insert(0, 0)  # add depot at the very beginning

    if os.path.exists(path_absolute + "/" + candidate_file_name):
        os.remove(path_absolute + "/" + candidate_file_name)

    if os.path.exists(path_absolute + "/" + par_file_name):
        os.remove(path_absolute + "/" + par_file_name)

    if os.path.exists(path_absolute + "/" + instance_file_name):
        os.remove(path_absolute + "/" + instance_file_name)
    if os.path.exists(path_absolute + "/" + tour_file_name):
        os.remove(path_absolute + "/" + tour_file_name)

    return tour


def get_lkh_results_dists(dists, symmetric=False):
    pid = os.getpid()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    pid = str(pid) + "_" + current_time
    par_file_name = "problem_tsp_temp" + pid + ".par"
    candidate_file_name = "cand_tsp_temp" + pid + ".txt"
    instance_file_name = "instance_tsp_temp" + pid + ".txt"
    tour_file_name = "tour_tsp_temp" + pid + ".txt"

    path_absolute = os.path.dirname(os.path.abspath(__file__))

    tourlength = len(dists)

    with open(path_absolute + "/" + par_file_name, "w") as file:
        file.write("PROBLEM_FILE = " + path_absolute + "/" + instance_file_name + "\n")
        file.write("TOUR_FILE = " + path_absolute + "/" + tour_file_name + "\n")
        file.write("CANDIDATE_SET_TYPE = ALPHA \n")

    with open(path_absolute + "/" + instance_file_name, "w") as file:
        file.write("NAME : " + instance_file_name[: len(instance_file_name) - 4] + "\n")
        file.write("COMMENT : random data, scaled \n")
        if symmetric:
            file.write("TYPE : TSP\n")
        else:
            file.write("TYPE : ATSP\n")
        file.write("DIMENSION : " + str(tourlength) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        file.write("EDGE_WEIGHT_SECTION\n")
        for i, dist in enumerate(dists):
            for d in dist:
                file.write(str(int(d * 1000)))
                file.write(" ")
            file.write("\n")

        file.write("EOF")

    subprocess.call(
        [LKH_SOLVER_PATH + "/LKH", path_absolute + "/" + par_file_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    with open(path_absolute + "/" + tour_file_name) as f:
        lines = f.readlines()

    length = lines[1]
    length = int(length.split()[-1]) / 1000
    lines = lines[6 : len(lines) - 2]
    tour = [int(line.split()[0]) - 1 for line in lines]

    if os.path.exists(path_absolute + "/" + candidate_file_name):
        os.remove(path_absolute + "/" + candidate_file_name)

    if os.path.exists(path_absolute + "/" + par_file_name):
        os.remove(path_absolute + "/" + par_file_name)

    if os.path.exists(path_absolute + "/" + instance_file_name):
        os.remove(path_absolute + "/" + instance_file_name)
    if os.path.exists(path_absolute + "/" + tour_file_name):
        os.remove(path_absolute + "/" + tour_file_name)

    return tour


def get_hgs_results_dists_cvrp(
    dists, demands, capacity, symmetric=False, TIME_LIMIT=None
):
    import hygese as hgs

    data = dict()
    data["distance_matrix"] = dists * 1000
    data["depot"] = 0
    data["demands"] = demands
    data["vehicle_capacity"] = capacity
    data["service_times"] = np.zeros(len(dists))

    if TIME_LIMIT is not None:
        ap = hgs.AlgorithmParameters(timeLimit=TIME_LIMIT)
    else:
        ap = hgs.AlgorithmParameters()

    hgs_solver = hgs.Solver(parameters=ap, verbose=False)

    result = hgs_solver.solve_cvrp(data)

    tour = [0]
    for elem in result.routes:
        tour.extend(elem)
        tour.append(0)

    return tour


def get_EA4OP_results_op(dists, prizes, max_length, coords=False):
    pid = os.getpid()
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    pid = str(pid) + "_" + current_time
    op_file_name = "problem_op_temp" + pid + ".oplib"

    if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/tmp"):
        os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/tmp")

    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/tmp" + "/" + op_file_name, "w"
    ) as file:
        ### GENERAL INSTANCE INFO
        file.write(f"NAME :{pid}" + "\n")
        file.write("COMMENT : none \n")
        file.write("TYPE : OP \n")
        file.write(f"DIMENSION : {len(prizes)}" + "\n")
        file.write(f"COST_LIMIT : {int(max_length* 100000)}" + "\n")

        if coords:
            file.write("EDGE_WEIGHT_TYPE : EUC_2D \n")
            file.write("NODE_COORD_SECTION \n")

            for i, node in enumerate(dists):
                file.write(str(i + 1))
                file.write(" ")
                file.write(str(node[0] * 100000))
                file.write(" ")
                file.write(str(node[1] * 100000))
                file.write("\n")

        else:
            file.write("EDGE_WEIGHT_TYPE: EXPLICIT \n")
            file.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX \n")
            ### SPECIFY EDGE WEIGHTS
            file.write("EDGE_WEIGHT_SECTION\n")
            for i, dist in enumerate(dists):
                for d in dist:
                    file.write(str(int(d * 100000)))
                    file.write(" ")
                file.write("\n")

        ### SPECIFY THE PRIZES
        file.write("NODE_SCORE_SECTION \n")
        for i, elem in enumerate(prizes):
            file.write(f"{i+1} {int(elem*100)}" + "\n")

        ### DEPOT INFO
        file.write("DEPOT_SECTION \n")
        file.write("1 \n")
        file.write("-1 \n")  # End depot section

        file.write("EOF")

    command = [
        "docker",
        "run",
        "-v",
        f"{os.path.dirname(os.path.abspath(__file__))}/tmp:/tmp",  # First volume mount
        "-v",
        f"{os.path.dirname(os.path.abspath(__file__))}"
        + "/tmp"
        + "/"
        + f"{op_file_name}:/tmp/{op_file_name}",  # Second volume mount
        "-it",
        "--rm",
        "op-solver",
        "opt",
        f"/tmp/{op_file_name}",  # Command inside container
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    if os.path.exists(
        os.path.dirname(os.path.abspath(__file__)) + "/tmp" + "/" + op_file_name
    ):
        os.remove(
            os.path.dirname(os.path.abspath(__file__)) + "/tmp" + "/" + op_file_name
        )

    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/tmp" + "/stats.json", "r"
    ) as f:
        lines = f.readlines()
        data = [json.loads(s) for s in lines]
        data = [x for x in data if x["prob"]["name"] == str(pid)]
        data = [x for x in data if x["env"] == "cp_heur_ea"]
        assert len(data) == 1, print(
            str(pid) + "Ensure that the docker container is up and running!"
        )
        data = data[0]

    tour = data["sol"]["cycle"]

    tour = [x - 1 for x in tour]
    tour.append(0)  ### close loop

    return tour
