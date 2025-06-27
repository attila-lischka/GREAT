import numpy as np

""" Code for baseline heuristics, provided by ChatGPT """


def farthest_insertion_tsp(distance_matrix):  # contribution by ChatGPT
    # Number of cities
    num_cities = len(distance_matrix)

    # Initialize tour with the first two farthest cities
    start_city = 0
    farthest_city = np.argmax(distance_matrix[start_city])
    tour = [start_city, farthest_city]
    unvisited = set(range(num_cities)) - set(tour)

    # Find the farthest city from the initial tour
    while unvisited:
        # Find the city in unvisited that is farthest from any city in the tour
        max_distance = -1
        for city in unvisited:
            min_distance_to_tour = min(distance_matrix[city][t] for t in tour)
            if min_distance_to_tour > max_distance:
                max_distance = min_distance_to_tour
                farthest_city = city

        # Insert the farthest city into the tour at the best position
        best_position = None
        best_increase = float("inf")

        for i in range(len(tour)):
            next_city = tour[(i + 1) % len(tour)]
            increase = (
                distance_matrix[tour[i]][farthest_city]
                + distance_matrix[farthest_city][next_city]
                - distance_matrix[tour[i]][next_city]
            )

            if increase < best_increase:
                best_increase = increase
                best_position = i + 1

        tour.insert(best_position, farthest_city)
        unvisited.remove(farthest_city)

    return tour


def nearest_insertion_tsp(distance_matrix):  # contribution by ChatGPT
    # Number of cities
    num_cities = len(distance_matrix)

    # Initialize tour with the first two nearest cities
    start_city = 0
    nearest_city = np.argmin(distance_matrix[start_city][1:]) + 1
    tour = [start_city, nearest_city]
    unvisited = set(range(num_cities)) - set(tour)

    # Insert the nearest city to the initial tour
    while unvisited:
        # Find the city in unvisited that is nearest to any city in the tour
        min_distance = float("inf")
        for city in unvisited:
            for t in tour:
                if distance_matrix[city][t] < min_distance:
                    min_distance = distance_matrix[city][t]
                    nearest_city = city
        # Insert the nearest city into the tour at the best position
        best_position = None
        best_increase = float("inf")

        for i in range(len(tour)):
            next_city = tour[(i + 1) % len(tour)]
            increase = (
                distance_matrix[tour[i]][nearest_city]
                + distance_matrix[nearest_city][next_city]
                - distance_matrix[tour[i]][next_city]
            )

            if increase < best_increase:
                best_increase = increase
                best_position = i + 1

        tour.insert(best_position, nearest_city)
        unvisited.remove(nearest_city)

    return tour


def nearest_neighbor_tsp(distance_matrix):
    # Number of cities
    num_cities = len(distance_matrix)

    # Initialize the tour starting with the first city
    start_city = 0
    tour = [start_city]
    unvisited = set(range(num_cities)) - {start_city}

    current_city = start_city

    while unvisited:
        # Find the nearest unvisited city
        nearest_city = min(
            unvisited, key=lambda city: distance_matrix[current_city][city]
        )

        # Add the nearest city to the tour
        tour.append(nearest_city)

        # Update the current city
        current_city = nearest_city

        # Remove the nearest city from the unvisited set
        unvisited.remove(nearest_city)

    return tour


def floyd_warshall(dist_matrix):
    n = dist_matrix.shape[0]
    # Copy the distance matrix so we don't overwrite the original
    dist = dist_matrix.copy()

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def dijkstra(dist_matrix):
    def neighbors(d, node):
        dis = d[node]
        return [(i, dis[i]) for i in range(len(dis))]

    import heapq

    n = dist_matrix.shape[0]

    rev_dists = dist_matrix.T

    distances = {i: float("inf") for i in range(n)}
    distances[0] = 0

    queue = [(0, 0)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # If we've already found a better path, skip this one
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors
        for neighbor, weight in neighbors(rev_dists, current_node):
            distance = current_distance + weight

            # If a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return np.array([distances[i] for i in range(len(distances))])


def _possible_moves(curr_length, return_distances, curr_tour):
    # compute moves that are still possible
    allowed_moves = set(range(len(return_distances))) - set(
        curr_tour
    )  # possbile to visit nodes that have not been visited yet
    curr_node = curr_tour[-1]
    return_distance = return_distances[curr_node]

    total_costs = return_distance + curr_length
    indices = np.where(
        (total_costs <= 1.0)
        & (np.isin(np.arange(total_costs.size), list(allowed_moves)))
    )[0]
    # returning is still possible and has not been visited yet

    return indices


def greedy_ratio_op(distances, prizes, max_length, xasy=False):
    # greedily go to the next node that has the best ratio of price and distance

    norm_distances = distances / max_length  # normalize given max_length

    if xasy:
        # the return cost to go from (i to j) is the cost that edge i,j has plus the shortest path cost to return to depot
        return_costs = dijkstra(norm_distances)  # shortest distance to each node
        return_costs = norm_distances + return_costs
    else:
        return_costs = norm_distances[:, 0]
        return_costs = (
            norm_distances + return_costs
        )  # compute cost for each node to visit other node and return to the depot

    tour = [0]
    tour_length = 0

    possible_moves = _possible_moves(tour_length, return_costs, tour)

    while len(possible_moves) > 0:
        # get prizes and distances given current state
        p = prizes[possible_moves]
        d = norm_distances[tour[-1], possible_moves]

        # compute ratio and chose highest one
        ratio = p / d
        greedy_index = np.argmax(ratio)
        move = possible_moves[greedy_index]

        # update tour
        tour_length += norm_distances[tour[-1], move]
        tour.append(move)
        if xasy:
            # necessary to recompute shortest distances to depot, since the previous shortest path potentially included a node that cannot be visited again.

            dists_masked = norm_distances.copy()
            # Set matching entries to inf
            dists_masked[:, [x for x in tour if x != 0]] = np.inf
            return_costs = dijkstra(dists_masked)  # shortest distance to each node
            return_costs = norm_distances + return_costs

        possible_moves = _possible_moves(tour_length, return_costs, tour)

    tour.append(0)
    tour_length += norm_distances[tour[-1], 0]

    return tour
