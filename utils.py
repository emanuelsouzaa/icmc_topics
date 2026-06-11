
import vrplib
import re
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_routes(
    routes: list[list[int]],
    coords,
    demands,
    optimal_cvrp,
    gap_opt,
    seed,
    instance_name,
    cost,
    K,
    save_path,
):
    depot = 0

    coords = np.asarray(coords)
    demands = np.asarray(demands)

    fig, ax = plt.subplots(figsize=(8, 8))

    is_customer = np.arange(len(demands)) != depot

    dmax = max(demands.max(), 1)
    sizes = 30 + (demands[is_customer] / dmax) * 220

    # Clientes

    ax.scatter(
        coords[is_customer, 0],
        coords[is_customer, 1],
        s=sizes,
        c="lightsteelblue",
        alpha=0.7,
        edgecolors="navy",
        linewidth=0.5,
        zorder=2
    )

    ax.scatter(
        coords[depot, 0],
        coords[depot, 1],
        s=320,
        c="crimson",
        marker="s",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        "Escola",
        (coords[depot, 0], coords[depot, 1]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="crimson",
    )

    colors = plt.cm.tab10.colors
    n_active = 0
    for r_idx, route in enumerate(routes):
        if len(route) <= 2:
            continue
        n_active += 1
        rx = [coords[i, 0] for i in route]
        ry = [coords[i, 1] for i in route]

        ax.plot(
            rx,
            ry,
            '-',
            color=colors[r_idx % 10],
            linewidth=1.6,
            alpha=0.85,
            zorder=3,
        )

    opt_str = str(optimal_cvrp) if optimal_cvrp else "\u2014"
    cost_str = f"{cost:.2f}" if cost else "\u2014"
    gap_str = f"{gap_opt:.2f}%" if gap_opt is not None else "\u2014"

    ax.set_title(
        f"{instance_name} \u2014 BRKGA seed {seed}\n"
        f"custo={cost_str}, \u00f3timo={opt_str} (gap {gap_str})\n"
        f"{len(routes)}/{K} rotas (K m\u00e1x)",
        fontsize=10,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="crimson", markeredgecolor="black",
               markersize=10, label="Escola (depósito)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="lightsteelblue", markeredgecolor="navy",
               markersize=9, label="Cliente (tamanho ∝ demanda)"),
    ]

    for r in range(n_active):
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[r % 10],
                linewidth=2.0,
                label=f"Rota {r + 1}"
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        fontsize=9,
        frameon=True,
        framealpha=0.9
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    plt.close(fig)


def unpacking_data(
    path_instance: str
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Extract data from an instance and calculate the Euclidean distance matrix.

    Parameters
    ----------
    data: dict

    Return
    -------
    N:  int
        Number of node
    K:  int
        Number of vehicles
    Q:  float
        Maximum load capacity
    d:  np.ndarray
        Demand vector
    D:  np.ndarray
        Distance matrix (N x N)
    """
    data = vrplib.read_instance(path_instance)

    N = data['dimension']
    K = int(data['name'].split('k')[1])
    Q = data['capacity']
    d = data['demand']
    node_coord = data['node_coord']

    opt = re.search(r"Optimal value:\s*([\d.]+)", data['comment'])
    if opt:
        optimal_value = float(opt.group(1))

    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            dist = np.hypot(
                node_coord[j][0] - node_coord[i][0],
                node_coord[j][1] - node_coord[i][1]
            )
            D[j, i] = D[i, j] = dist

    return N, K, Q, d, D, optimal_value, node_coord
