import numpy as np
from numpy.typing import NDArray


def two_regret_insertion(
    N: int,
    K: int,
    Q: float,
    d: NDArray[np.float64],
    D: NDArray[np.float64]
) -> list[list[int]]:
    """
    Construct an initial solution for the CVRP using the 2-Regret heuristic.

    Parameters
    ------------
    N:  int
        Number of nodes.
    K:  int
        Number of vehicles.
    Q:  float
        Maximum load capacity of each vehicle.
    d:  np.ndarray
        Demand vector.
    D:  np.ndarray
        Distance matrix.

    Returns:
    ---------
    routes: list[list[int]]
            A list of routes. Each route starts and ends at depot 0.
    Raises
    ------
    ValueError
        If no feasible insertion exists for a remaining customer.
    """

    routes = [[0, 0] for _ in range(K)]
    capacity = N * [0.0]
    visited_nodes = {0}

    while len(visited_nodes) < N:

        node_data = {}

        for node in range(1, N):
            if node in visited_nodes:
                continue

            best_by_route = {}

            for k in range(K):

                if capacity[k] + d[node] <= Q:
                    route = routes[k]

                    best_cost_in_k = float('inf')
                    best_pos_in_k = -1

                    for i in range(len(route) - 1):
                        a, b = route[i], route[i + 1]
                        cost = D[a][node] + D[node][b] - D[a][b]

                        if cost < best_cost_in_k:
                            best_cost_in_k = cost
                            best_pos_in_k = i + 1

                    if best_pos_in_k != -1:
                        best_by_route[k] = {
                            "cost": best_cost_in_k,
                            "route_idx": k,
                            "pos": best_pos_in_k
                        }

            if not best_by_route:
                continue

            sorted_route_options = sorted(
                best_by_route.values(),
                key=lambda x: x["cost"]
            )

            best_opt = sorted_route_options[0]

            if len(sorted_route_options) > 1:
                regret = sorted_route_options[1]["cost"] - best_opt["cost"]
            else:
                regret = 1e9 + best_opt["cost"]

            node_data[node] = {"regret": regret, "best_info": best_opt}

        if not node_data:
            raise ValueError(
                "It was not possible to insert all nodes with "
                "the current constraints."
            )

        best_node = max(node_data, key=lambda n: node_data[n]["regret"])

        info = node_data[best_node]["best_info"]
        k_idx = info["route_idx"]
        pos = info["pos"]

        routes[k_idx].insert(pos, int(best_node))
        capacity[k_idx] += float(d[best_node])
        visited_nodes.add(best_node)

    return routes
