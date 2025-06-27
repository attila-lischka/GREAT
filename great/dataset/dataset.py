import os
import random

import numpy as np
import torch
from joblib import Parallel, delayed
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from great.utils.constants import DATA_PATH, TEST_DATA_LOAD_PATH

from .data_object import (
    get_asymmetric_data_object,
    get_data_object,
    init_distances_fast,
)


class TestDataset(InMemoryDataset):
    ### a wrapper class to import the concorde test dataset
    def __init__(
        self,
        transform=None,
        pre_transform=None,
    ):
        super(TestDataset, self).__init__(DATA_PATH, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # List of processed file names
        filename = "tsp100_concorde_EUC"
        filename += ".pt"

        return [filename]

    def get_gurobi_tours(self):
        sol_file_name = "EUC_100_solution.txt"

        # Open the file in write mode
        with open(os.path.join(TEST_DATA_LOAD_PATH, sol_file_name), "r") as file:
            lines = file.readlines()

        tours = []

        for line in lines:
            _l = line.split()
            _l = [int(x) for x in _l]
            tours.append(_l)

        return tours

    def process(self):
        coordinates = []
        concorde_tours = []
        entries = os.listdir(TEST_DATA_LOAD_PATH)
        files = [
            entry
            for entry in entries
            if os.path.isfile(os.path.join(TEST_DATA_LOAD_PATH, entry))
        ]
        files = [f for f in files if ".DS_" not in f]
        file = [f for f in files if "tsp100_test_concorde_EUC.txt" in f][0]
        abs_file = TEST_DATA_LOAD_PATH + "/" + file
        with open(abs_file) as f:
            lines = f.readlines()

        lines = [line.split() for line in lines]
        for line in lines:
            c = line[:200]
            t = line[201:-1]
            c = [float(entry) for entry in c]
            t = [int(entry) - 1 for entry in t]
            coord = []
            for i in range(0, len(c), 2):
                coord.append([c[i], c[i + 1]])
            coordinates.append(coord)
            concorde_tours.append(t)

        gurobi_tours = (
            self.get_gurobi_tours()
        )  ### for some reason, the Concorde tours are suboptimal so use gurobi instead

        coordinates = np.array(coordinates)
        concorde_tours = np.array(concorde_tours)
        gurobi_tours = np.array(gurobi_tours)

        processed_data_list = Parallel(n_jobs=16, verbose=10)(
            delayed(get_data_object)(coords, False, False, tour)
            for coords, tour in zip(coordinates, gurobi_tours)
        )
        # safe data
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices), self.processed_paths[0])


class GREATRLDataset(InMemoryDataset):
    ### same as GREATDataset but without labels and therefore data is also not saved but generated on the fly to save storage
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        data_dist="EUC",
        problem="TSP",
        dataset_size=1000,
        instance_size=50,
        seed=0,
        save_raw_data=False,
        save_processed_data=False,  # generally we don't do this because it needs lot of storage
    ):
        assert data_dist in ["EUC", "SYM", "XASY", "TMAT", "MIX"]
        assert problem in ["TSP", "CVRP", "OP"]
        self.seed = seed
        self.dataset_size = dataset_size
        self.instance_size = instance_size
        self.data_dist = data_dist
        self.problem = problem
        self.save_raw_data = save_raw_data
        self.save_processed_data = save_processed_data

        super(GREATRLDataset, self).__init__(DATA_PATH, transform, pre_transform)

        if self.save_processed_data and os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.generate_data()

    @property
    def raw_file_names(self):
        # We do not necessariry save the raw coordinates. Only if self.save_raw_data = True, e.g. we might want to keep the coordinates for val dataset
        # List of processed file names
        filename = f"{self.problem}{self.instance_size}_dataset{self.dataset_size}_RL_"
        filename += self.data_dist
        filename += str(self.seed)
        filename += ".txt"
        return [filename]

    @property
    def processed_file_names(self):
        # List of processed file names
        filename = f"{self.problem}{self.instance_size}_dataset{self.dataset_size}_RL_"
        filename += self.data_dist
        filename += str(self.seed)
        filename += ".pt"

        return [filename]

    def generate_symmetric_matrix(self, size):
        # Generate a random matrix with values between 0 and 1
        A = np.random.rand(size, size)

        # Make the matrix symmetric
        A = np.triu(A)
        symmetric_matrix = A + A.T

        # Set the diagonal entries to 0
        np.fill_diagonal(symmetric_matrix, 0)

        return symmetric_matrix

    def generate_tmat_matrix(self, size):
        problems = torch.randint(low=1, high=1000000 + 1, size=(1, size, size))
        problems[:, torch.arange(size), torch.arange(size)] = 0
        while True:
            old_problems = problems.clone()
            problems, _ = (
                problems[:, :, None, :] + problems[:, None, :, :].transpose(2, 3)
            ).min(dim=3)
            if (problems == old_problems).all():
                break
        problems = problems.squeeze()
        problems = problems / problems.max()

        return problems.cpu().numpy()

    def generate_data(self):
        # set the seeds for the data generation
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.problem == "CVRP":  # sample customer demands
            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}
            # capacity of the vehicle in the CVRP instance
            instance_features = CAPACITIES[self.instance_size]
            # demands of the customers in the CVRP instance
            node_features = torch.randint(
                1, 10, size=(self.dataset_size, self.instance_size)
            ).numpy()
            node_features = (
                node_features / instance_features
            )  # normalize given the capacity
            # Adding of -1 demands for the depot
            depot_values = -1 * np.ones(
                (self.dataset_size, 1)
            )  # Shape (n, 1), represents the demand of -1 of the depot
            node_features = np.hstack((depot_values, node_features))
            distances = self.get_distances()

        elif self.problem == "OP":  # sample prizes
            # From https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/op/problem_op.py
            MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
            prizes = (
                1 + torch.randint(0, 100, size=(self.dataset_size, self.instance_size))
            ) / 100.0
            instance_features = MAX_LENGTHS[self.instance_size]
            instance_features = np.full(self.dataset_size, instance_features)
            if self.data_dist == "XASY":
                # The average TSP length for an XASY instance with 100 nodes is ~2, so an OP with max_length = 4 is basically always solving the TSP.
                # a max_length of 0.4 does not allow us to travel to virtually all nodes, however, so we shrink the maximum_length by a factor of 10
                instance_features = instance_features / 10
            elif self.data_dist == "MIX":
                fraction = self.dataset_size // 3
                remainder_fraction = self.dataset_size - 2 * fraction
                # the last #remainder_fraction many instances of the dataset in the mixed case are XASY instances so they need rescaling
                instance_features[-remainder_fraction:] /= 10

            node_features = prizes.numpy()
            depot_values = np.zeros((self.dataset_size, 1))  # Shape (n, 1)
            node_features = np.hstack((depot_values, node_features))
            distances = self.get_distances()

            # so far, the node features of a OP instance encode the prize
            # we also want to have another node feature that represents the distance of a node to the depot
            # (this is important for the model when it want to determine if it can still go back after visiting a node)
            if self.data_dist == "EUC":
                temp_distances = np.array(
                    [init_distances_fast(coords) for coords in distances]
                )
            else:
                temp_distances = distances
            temp_distances = (
                temp_distances / instance_features[:, np.newaxis, np.newaxis]
            )

            node_features = node_features[..., np.newaxis]
            depot_distances = temp_distances[:, :, 0]
            depot_distances = depot_distances[..., np.newaxis]

            node_features = np.concatenate(
                (depot_distances, node_features), axis=-1
            )  # last dim is the prize
        else:  # normal TSP
            node_features = [None] * self.dataset_size
            instance_features = None
            distances = self.get_distances()

        if self.problem == "TSP" and self.data_dist == "EUC":
            processed_data_list = Parallel(n_jobs=16, verbose=10)(
                delayed(get_data_object)(coords, False, False) for coords in distances
            )

        else:
            # Routing for non-EUC always has asymmetric edges, either because of distance or because of node features
            # E.g., if edge(i,j) contains the node feature of edge j (e.g. the demand of j) than edge(j,i) is different as i might have a different demand than j
            if self.data_dist == "EUC":  # transform coordinates in distance matrices
                coordinates = distances
                distances = np.array(
                    [init_distances_fast(coords) for coords in distances]
                )
            else:
                coordinates = [None] * len(distances)

            if self.problem == "OP":  # OP is a special case
                # normalize distances based on the maximum allowed tour length
                distances_normalized = (
                    distances / instance_features[:, np.newaxis, np.newaxis]
                )
                distances = distances[..., np.newaxis]
                distances_normalized = distances_normalized[..., np.newaxis]
                distances = np.concatenate((distances, distances_normalized), axis=-1)

                processed_data_list = Parallel(n_jobs=16, verbose=10)(
                    delayed(get_asymmetric_data_object)(
                        dists, node_feat, i_feat, coords
                    )
                    for dists, node_feat, i_feat, coords in zip(
                        distances, node_features, instance_features, coordinates
                    )
                )

            else:
                processed_data_list = Parallel(n_jobs=-1, verbose=10)(
                    delayed(get_asymmetric_data_object)(
                        dists, node_feat, instance_features, coords
                    )
                    for dists, node_feat, coords in zip(
                        distances, node_features, coordinates
                    )
                )

        if self.data_dist == "MIX":
            random.shuffle(processed_data_list)
        # save data
        data, slices = self.collate(processed_data_list)

        if self.save_processed_data:
            torch.save((data, slices), self.processed_paths[0])

        return data, slices

    def get_distances(self):
        # set the seeds for the data generation
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.problem == "TSP":
            sample_size = self.instance_size
        else:  # for OP and CVRP, we have an additional depot
            sample_size = self.instance_size + 1

        # sample coordinates given the dataset specifications
        if (
            self.data_dist == "XASY"
        ):  ### eXtreme ASYmmetric, directions are completely uncorrelated
            # sample random edge distances. Diagonal entries should be zero but are random in this case but it does not matter as
            # we just ignore self loops in the further processing anyway.
            edge_distances = np.random.uniform(
                size=(self.dataset_size, sample_size, sample_size)
            )
            if self.save_raw_data:  # save raw coordinates
                array_2d = edge_distances.reshape(-1, edge_distances.shape[-1])
                np.savetxt(self.raw_paths[0], array_2d, delimiter=" ", fmt="%.8f")

        elif self.data_dist == "EUC":  # euclidean data
            coordinates = np.random.uniform(size=(self.dataset_size, sample_size, 2))

            if self.save_raw_data:  # save raw coordinates
                self.coordinates = coordinates
                with open(self.raw_paths[0], "w") as f:
                    f.write(" ".join(map(str, coordinates)))

            edge_distances = coordinates

        elif (
            self.data_dist == "SYM"
        ):  # SYMmetric problem, but without "euclidean background"
            edge_distances = np.zeros((self.dataset_size, sample_size, sample_size))

            for i in range(self.dataset_size):
                edge_distances[i] = self.generate_symmetric_matrix(sample_size)

            if self.save_raw_data:  # save raw coordinates
                with open(self.raw_paths[0], "w") as f:
                    f.write(" ".join(map(str, edge_distances)))

        elif (
            self.data_dist == "TMAT"
        ):  # asymmetric problem where the triangle inequality holds
            edge_distances = np.zeros((self.dataset_size, sample_size, sample_size))
            for i in tqdm(range(self.dataset_size)):
                edge_distances[i] = self.generate_tmat_matrix(sample_size)

            if self.save_raw_data:  # save raw coordinates
                with open(self.raw_paths[0], "w") as f:
                    f.write(" ".join(map(str, edge_distances)))

        elif self.data_dist == "MIX":
            fraction = self.dataset_size // 3
            remainder_fraction = self.dataset_size - 2 * fraction

            # EUC
            coords = np.random.uniform(size=(fraction, sample_size, 2))
            euc_distances = np.array([init_distances_fast(c) for c in coords])
            # TMAT
            tmat_distances = np.zeros((fraction, sample_size, sample_size))
            for i in tqdm(range(fraction)):
                tmat_distances[i] = self.generate_tmat_matrix(sample_size)

            # XASY
            xasy_distances = np.random.uniform(
                size=(remainder_fraction, sample_size, sample_size)
            )
            # combine the different data
            edge_distances = np.concatenate(
                (euc_distances, tmat_distances, xasy_distances), axis=0
            )

            if self.save_raw_data:  # save raw coordinates
                with open(self.raw_paths[0], "w") as f:
                    f.write(" ".join(map(str, edge_distances)))

        else:
            assert False, "unknown data distribution " + self.data_dist

        return edge_distances


class GREATTSPDataset(InMemoryDataset):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        dataset_size=1000,
        tsp_size=50,
        seed=0,
        save_raw_data=False,
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.tsp_size = tsp_size
        self.save_raw_data = save_raw_data

        super(GREATTSPDataset, self).__init__(DATA_PATH, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We do not necessariry safe the raw coordinates. Only if self.save_raw_data = True, e.g. we might want to keep the coordinates for val dataset
        # List of processed file names
        filename = f"tsp{self.tsp_size}_dataset{self.dataset_size}_"
        filename += str(self.seed)
        filename += ".txt"
        return [filename]

    @property
    def processed_file_names(self):
        # List of processed file names
        filename = f"tsp{self.tsp_size}_dataset{self.dataset_size}_"
        filename += str(self.seed)
        filename += ".pt"

        return [filename]

    def download(self):
        # Not implemented as we are generating data
        pass

    def process(self):
        # set the seeds for the data generation
        random.seed(self.seed)
        np.random.seed(self.seed)
        # sample coordinates given the dataset specifications
        coordinates = np.random.uniform(size=(self.dataset_size, self.tsp_size, 2))

        if self.save_raw_data:  # save raw coordinates
            with open(self.raw_paths[0], "w") as f:
                f.write(" ".join(map(str, coordinates)))

        # get a list of data objects
        processed_data_list = Parallel(n_jobs=16, verbose=10)(
            delayed(get_data_object)(coords, True, True) for coords in coordinates
        )
        # save data
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices), self.processed_paths[0])


class GLOPBenchmarkDataset(InMemoryDataset):
    ### same as GREATDataset but without labels and therefore data is also not saved but generated on the fly to save storage
    def __init__(
        self,
        dists,
        transform=None,
        pre_transform=None,
    ):
        self.dists = dists
        self.dataset_size = len(dists)
        self.size = len(dists[0])

        super(GLOPBenchmarkDataset, self).__init__(DATA_PATH, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.generate_data()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # List of processed file names
        # List of processed file names
        filename = f"GLOP_dataset{self.dataset_size}_{self.size}"
        filename += ".pt"

        return [filename]

    def generate_data(self):
        # set the seeds for the data generation
        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        node_features = [None] * self.dataset_size
        instance_features = None
        distances = self.dists.numpy()

        # normalize distances such that the biggest element is 1 in each distance matrix
        max_vals = np.max(distances, axis=(1, 2), keepdims=True)
        # Avoid division by zero (optional, in case any matrix is all zeros)
        max_vals[max_vals == 0] = 1
        # Normalize each matrix
        distances = distances / max_vals

        coordinates = [None] * len(distances)

        processed_data_list = Parallel(n_jobs=-1, verbose=10)(
            delayed(get_asymmetric_data_object)(
                dists, node_feat, instance_features, coords
            )
            for dists, node_feat, coords in zip(distances, node_features, coordinates)
        )

        data, slices = self.collate(processed_data_list)

        torch.save((data, slices), self.processed_paths[0])

        return data, slices
