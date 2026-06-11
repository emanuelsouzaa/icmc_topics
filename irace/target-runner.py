#!/usr/bin/env python3

import os
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(PARENT_DIR, "configs", "config.conf")

sys.path.insert(0, PARENT_DIR)

from brkga_mp_ipr.types_io import load_configuration  # noqa: E402
from brkga_mp_ipr.algorithm import BrkgaMpIpr  # noqa: E402
from brkga_mp_ipr.enums import Sense, BiasFunctionType  # noqa: E402
from decoder import VRPDecoder  # noqa: E402
from utils import unpacking_data  # noqa: E402


def run(instance_path, seed, population_size, elite_percentage,
        mutants_percentage, num_elite_parents, total_parents,
        bias_type, num_independent_populations, exchange_interval,
        reset_interval, n_runs=3, max_time=60.0):

    N, K, Q, d, D, _, _ = unpacking_data(
        os.path.join(PARENT_DIR, instance_path)
    )

    brkga_params, _ = load_configuration(CONFIG_PATH)

    brkga_params.population_size = population_size
    brkga_params.elite_percentage = elite_percentage
    brkga_params.mutants_percentage = mutants_percentage
    brkga_params.num_elite_parents = num_elite_parents
    brkga_params.total_parents = total_parents
    brkga_params.bias_type = getattr(BiasFunctionType, bias_type)
    brkga_params.num_independent_populations = num_independent_populations
    brkga_params.exchange_interval = exchange_interval
    brkga_params.reset_interval = reset_interval

    decoder = VRPDecoder(K, N, Q, d, D)

    costs = []

    for run_idx in range(n_runs):
        brkga = BrkgaMpIpr(
            decoder=decoder,
            sense=Sense.MINIMIZE,
            seed=seed + run_idx,
            chromosome_size=N - 1,
            params=brkga_params
        )
        brkga.initialize()

        best_cost = float('inf')
        t0 = time.time()

        while time.time() - t0 < max_time:
            brkga.evolve()
            routes_best = decoder.get_routes(brkga.get_best_chromosome())
            routes_best = decoder.two_swap(routes_best)
            cost = decoder.total_cost(routes_best)

            if cost < best_cost - 1e-6:
                best_cost = cost

        costs.append(best_cost)

    return np.mean(costs)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("bound", type=str)
    parser.add_argument("instance", type=str)
    parser.add_argument("--population_size", type=int, required=True)
    parser.add_argument("--elite_percentage", type=float, required=True)
    parser.add_argument("--mutants_percentage", type=float, required=True)
    parser.add_argument("--num_elite_parents", type=int, required=True)
    parser.add_argument("--total_parents", type=int, required=True)
    parser.add_argument("--bias_type", type=str, required=True)
    parser.add_argument(
        "--num_independent_populations",
        type=int,
        required=True
    )
    parser.add_argument("--exchange_interval", type=int, required=True)
    parser.add_argument("--reset_interval", type=int, required=True)

    args = parser.parse_args()

    cost = run(
        instance_path=args.instance,
        seed=args.seed,
        population_size=args.population_size,
        elite_percentage=args.elite_percentage,
        mutants_percentage=args.mutants_percentage,
        num_elite_parents=args.num_elite_parents,
        total_parents=args.total_parents,
        bias_type=args.bias_type,
        num_independent_populations=args.num_independent_populations,
        exchange_interval=args.exchange_interval,
        reset_interval=args.reset_interval
    )

    print(cost)
