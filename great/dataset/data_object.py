import math

import numpy as np
import torch
from torch_geometric.data import Data


def init_distances(nodes):
    """
    Function that return a distance matrix based on the Euclidean distances of the passed nodes.
    nodes: coordinates of the nodes in for which we want to compute the distances. Dimension: n * 2
    Returns: An n * n matrix where entry (i,j) corresponds to the Euclidean distance between node i and j.
    """
    size = len(nodes)
    distance = [[0 for _ in range(size)] for _ in range(size)]
    for j in range(size):
        for k in range(j, size):
            x1 = nodes[j][0]
            x2 = nodes[k][0]
            y1 = nodes[j][1]
            y2 = nodes[k][1]
            distance[j][k] = distance[k][j] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


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


def organize_distances(distances):
    """
    Helper function that sorts the distances.
    distances: A n * n distance matrix where entry (i,j) corresponds to the Euclidean distance between node i and j.
    Returns: A list of lists where the ith list contains tuples indicating the nearest neighbors of node i.
             The tuple contains the index of the neighbor and the actual distance to node i. The tuples are sorted in
             ascending order based on the distance
    """
    results = []
    for distance in distances:
        ordered = sorted(range(len(distance)), key=lambda k: distance[k])
        results.append([[i, distance[i]] for i in ordered])
    return results


def get_augmented_edge_features(dists, org_dists):
    res = []
    size = len(dists)
    for i in range(size):
        _temp = []
        for j in range(size):
            _feature = []
            if i == j:
                _temp.append([0, 0])
                continue
            val = (dists[i][j] - org_dists[i][1][1]) / (
                org_dists[i][size - 1][1] - org_dists[i][1][1]
            )  # edge length compared to minimal edge length originating in this node
            _feature.append(val)
            temp_sum = sum([org_dists[i][k][1] for k in range(1, size)])
            val = (dists[i][j] - temp_sum / (size - 1)) / (
                org_dists[i][size - 1][1] - org_dists[i][1][1]
            )  # edge length compared to average edge length originating in this node
            _feature.append(val)
            _temp.append(_feature)

        res.append(_temp)

    return res


def get_EUC_TSP_cost(coordinates: np.ndarray, tour: list) -> float:
    """
    Helper function that tells the TSP cost of an Euclidean TSP
    """

    def euclidean_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    total_distance = 0
    for i in range(len(tour)):
        # Get the current point and the next point (wrap around at the end)
        current_point = coordinates[tour[i]]
        next_point = coordinates[tour[(i + 1) % len(tour)]]

        # Add the distance between the points
        total_distance += euclidean_distance(current_point, next_point)

    return total_distance


def get_data_object(
    coordinates: np.ndarray, cost=False, tour=False, gurobi_tour=None
) -> Data:
    """
    Given a np array of nodes coordinates, creates a pytorch geometric data object
    """
    n = len(coordinates)
    x = torch.tensor(
        coordinates,
        dtype=torch.float32,  # save for Pointerformer model
    )
    d = init_distances_fast(coordinates)

    _d = organize_distances(d)
    aug_feat = get_augmented_edge_features_asy(d, _d, _d)

    edge_index1 = []
    edge_index2 = []
    edge_attr = []
    edge_target = []

    if tour or cost:
        from great.utils.tour_wrapper import get_lkh_results

        lkh_tour = get_lkh_results(coordinates)
        tour_edges = {
            (lkh_tour[i], lkh_tour[i + 1]) for i in range(len(lkh_tour) - 1)
        } | {(lkh_tour[i + 1], lkh_tour[i]) for i in range(len(lkh_tour) - 1)}
        tour_edges.add((lkh_tour[0], lkh_tour[-1]))
        tour_edges.add((lkh_tour[-1], lkh_tour[0]))

        TSP_cost = get_EUC_TSP_cost(coordinates, lkh_tour)

    if gurobi_tour is not None:
        tour = True
        cost = True

        tour_edges = {
            (gurobi_tour[i], gurobi_tour[i + 1]) for i in range(len(gurobi_tour) - 1)
        } | {(gurobi_tour[i + 1], gurobi_tour[i]) for i in range(len(gurobi_tour) - 1)}
        tour_edges.add((gurobi_tour[0], gurobi_tour[-1]))
        tour_edges.add((gurobi_tour[-1], gurobi_tour[0]))

        TSP_cost = get_EUC_TSP_cost(coordinates, gurobi_tour)

    for i in range(n):
        for j in range(i + 1, n):
            edge_index1.extend([i, j])
            edge_index2.extend([j, i])

            _first_edge = [d[i, j]] + aug_feat[i][j]
            _second_edge = [d[j, i]] + aug_feat[j][i]
            edge_attr.extend([_first_edge, _second_edge])

            if tour:
                if (i, j) in tour_edges:
                    edge_target.append(1)
                else:
                    edge_target.append(0)
                if (j, i) in tour_edges:
                    edge_target.append(1)
                else:
                    edge_target.append(0)

    edges = np.array(
        [edge_index1, edge_index2]
    )  ### converting to numpy first and then to torch is faster for some reason
    edge_attr = np.array(edge_attr)
    edge_target = np.array(edge_target)
    args_dict = dict()
    args_dict["edge_attr"] = torch.from_numpy(edge_attr).float()
    args_dict["edge_index"] = torch.tensor(edges)

    if tour:
        args_dict["edge_target"] = torch.from_numpy(edge_target).long()

    if cost:
        args_dict["instance_target"] = torch.tensor(TSP_cost)

    return Data(x=x, num_nodes=n, **args_dict)


def get_augmented_edge_features_asy(dists, org_dists_out, org_dists_in):
    """For each edge (i,j) we want to get the shortest edge (i,k) and the average (i, :)
    aswell as the shortest (k,j) and average (:,j)
    """
    res = []
    size = len(dists)
    for i in range(size):
        _temp = []
        for j in range(size):
            _feature = []
            if i == j:
                _temp.append([0, 0, 0, 0])
                continue
            val = (dists[i][j] - org_dists_out[i][1][1]) / (
                org_dists_out[i][size - 1][1] - org_dists_out[i][1][1]
            )  # edge length compared to minimal edge length originating in this node
            _feature.append(val)
            temp_sum = sum([org_dists_out[i][k][1] for k in range(1, size)])
            val = (dists[i][j] - temp_sum / (size - 1)) / (
                org_dists_out[i][size - 1][1] - org_dists_out[i][1][1]
            )  # edge length compared to average edge length originating in this node
            _feature.append(val)
            val = (dists[j][i] - org_dists_in[j][1][1]) / (
                org_dists_in[j][size - 1][1] - org_dists_in[j][1][1]
            )  # edge length compared to minimal edge length originating in this node
            _feature.append(val)
            temp_sum = sum([org_dists_in[j][k][1] for k in range(1, size)])
            val = (dists[j][i] - temp_sum / (size - 1)) / (
                org_dists_in[j][size - 1][1] - org_dists_in[j][1][1]
            )  # edge length compared to average edge length originating in this node
            _feature.append(val)
            _temp.append(_feature)

        res.append(_temp)
    return res


def get_asymmetric_data_object(
    edge_dists, node_features, instance_features, coordinates
) -> Data:
    # some assertions
    additional_edge_features = None
    if edge_dists.ndim == 3:
        additional_edge_features = edge_dists[:, :, 1:]
        edge_dists = edge_dists[
            :, :, 0
        ]  # by our convention, the first entry is always the actual distance

    edge_dists = np.array(
        edge_dists
    )  # for some reason, if we do not do this, we sometimes run into an "ValueError: assignment destination is read-only" later

    assert len(edge_dists) == len(edge_dists[0])
    if node_features is not None:
        assert len(edge_dists) == len(node_features)

    n = len(edge_dists)
    if coordinates is not None:
        x = torch.tensor(
            coordinates,
            dtype=torch.float32,  # save for Pointerformer model
        )
    else:
        x = torch.arange(0, n).float()

    for i in range(n):
        edge_dists[i, i] = 0

    _d_out = organize_distances(edge_dists)
    _d_in = organize_distances(np.transpose(edge_dists))
    aug_feat = get_augmented_edge_features_asy(edge_dists, _d_out, _d_in)

    edge_index1 = []
    edge_index2 = []
    edge_attr = []

    for i in range(n):
        for j in range(i + 1, n):
            edge_index1.extend([i, j])
            edge_index2.extend([j, i])

            # for each edge encode the distance in the augmented features ("how long is the edge compared to other edges?")
            _first_edge = [edge_dists[i, j]] + aug_feat[i][j]
            _second_edge = [edge_dists[j, i]] + aug_feat[j][i]

            if additional_edge_features is not None:
                _first_edge.extend(additional_edge_features[i, j])
                _second_edge.extend(additional_edge_features[j, i])

            # in case there are node features: encode feature of node j for the edge (i,j)
            # this makes sense, since e.g. in CVRP, taking edge (i,j) means that we have to have capacity for the demand of customer j
            if node_features is not None:
                if np.isscalar(node_features[j]):
                    _first_edge.append(node_features[j])
                else:
                    _first_edge.extend(node_features[j])

                if np.isscalar(node_features[i]):
                    _second_edge.append(node_features[i])
                else:
                    _second_edge.extend(node_features[i])
            edge_attr.extend([_first_edge, _second_edge])

    edges = np.array(
        [edge_index1, edge_index2]
    )  ### converting to numpy first and then to torch is faster for some reason
    edge_attr = np.array(edge_attr)
    args_dict = dict()
    args_dict["edge_attr"] = torch.from_numpy(edge_attr).float()
    args_dict["edge_index"] = torch.tensor(edges)

    if instance_features is not None:
        args_dict["instance_feature"] = torch.tensor(
            instance_features, dtype=torch.float32
        )

    return Data(x=x, num_nodes=n, **args_dict)
