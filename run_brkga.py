import os
import numpy as np
import csv
import time
import json
from pathlib import Path
from decoder import VRPDecoder
from utils import unpacking_data, plot_routes
from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from brkga_mp_ipr.types_io import load_configuration


def run_brkga(
    instance_path: str,
    params_path: str,
    max_time: int = 60,
    seeds: list = None,
    output_dir: str = "results_scenario_1_BRKGA",
    scenario: str = "C1",
):

    assert seeds is not None, "Missing seeds"

    max_time_min = max_time / 60

    os.makedirs(output_dir, exist_ok=True)

    N, K, Q, d, D, optimal, node_coord = unpacking_data(instance_path)
    brkga_params, _ = load_configuration(params_path)
    decoder = VRPDecoder(K, N, Q, d, D)

    instance_name = Path(instance_path).stem
    n_runs = len(seeds)

    print(f"\n{'='*70}")
    print(f"Instância: {instance_name} | N={N} | K={K} | Ótimo={optimal}")
    print(f"Rodadas: {n_runs} | Tempo máximo: {max_time_min:2f}m")
    print(f"\n{'='*70}")

    all_costs = []
    all_curves = []
    best_overall_costs = float("inf")
    best_overall_routes = None

    # results file
    csv_path = os.path.join(output_dir, "resultados_by_seed.csv")
    csv_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as csv_file:

        field_names = [
            "cenario",
            "metodo",
            "instancia",
            "seed",
            "custo_total",
            "iteracoes",
            "arquivo_solucao",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        if not csv_exists:
            writer.writeheader()

        for run_id, seed in enumerate(seeds):
            brkga_vrp = BrkgaMpIpr(
                decoder=decoder,
                sense=Sense.MINIMIZE,
                seed=seed,
                chromosome_size=N - 1,
                params=brkga_params,
            )

            brkga_vrp.initialize()

            best_cost = float("inf")
            curve = []
            best_iter = 0
            best_routes = None
            it = 0
            t0 = time.time()

            while time.time() - t0 < max_time:

                brkga_vrp.evolve()
                it += 1
                routes_best = decoder.get_routes(
                    brkga_vrp.get_best_chromosome()
                )
                routes_best_local = decoder.two_swap(routes_best)
                cost = decoder.total_cost(routes_best_local)
                curve.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_iter = it
                    best_routes = routes_best_local

            all_costs.append(best_cost)
            all_curves.append(curve)

            json_filename = (
                f"sol_{scenario}_BRKGA_{instance_name}_seed_{seed}.json"
            )
            json_path = os.path.join(output_dir, json_filename)
            sol_data = {
                "custo_total": best_cost,
                "rotas": [[0] + r + [0] for r in best_routes],
                "coordenadas": {
                    f"{i}": node_coord[i].tolist() for i in range(N)
                },
            }

            with open(json_path, "w") as f:
                json.dump(sol_data, f, indent=2)

            if best_cost < best_overall_costs:
                best_overall_costs = best_cost
                best_overall_routes = best_routes

            gap_str = (
                f" | Gap: {((best_cost - optimal) / optimal * 100):.2f}%"
                if optimal
                else ""
            )
            print(
                f"Rodada {(run_id + 1):2d} | Seed {seed:4d} |"
                f"Custo: {best_cost:.2f} | Iter: {best_iter} {gap_str}"
            )

            writer.writerow(
                {
                    "cenario": scenario,
                    "metodo": "BRKGA",
                    "instancia": instance_name,
                    "seed": seed,
                    "custo_total": round(best_cost, 4),
                    "iteracoes": it,
                    "arquivo_solucao": json_filename,
                }
            )

    mean_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    best_cost = np.min(all_costs)
    best_seed = seeds[np.argmin(all_costs)]
    worst_cost = np.max(all_costs)
    gap_best = ((best_cost - optimal) / optimal * 100) if optimal else None
    gap_mean = ((mean_cost - optimal) / optimal * 100) if optimal else None
    dispersion = (
        ((worst_cost - best_cost) / best_cost * 100) if best_cost > 0 else None
    )

    print(f"\n{'='*70}")
    print(f"Melhor:           {best_cost:.2f}")
    print(f"Média:            {mean_cost:.2f}")
    print(f"Desvio padrão:    {std_cost:.2f}")
    print(f"Pior:             {worst_cost:.2f}")
    print(f"Dispersão:        {dispersion:.2f}%")

    if optimal:
        print(f"GAP(melhor):      {gap_best:.2f}")
        print(f"GAP(médio) :      {gap_mean:.2f}")

    print(f"\n{'='*70}")

    # Guardando resultados finais

    csv_path_final = os.path.join(output_dir, "resultados_final.csv")
    csv_path_final_exists = os.path.exists(csv_path_final)

    with open(csv_path_final, "a", newline="") as csv_file:

        field_names_final = [
            "cenario",
            "metodo",
            "instancia",
            "melhor custo",
            "gap melhor",
            "pior custo",
            "dispersao"
        ]

        writer_final = csv.DictWriter(csv_file, fieldnames=field_names_final)

        if not csv_path_final_exists:
            writer_final.writeheader()

        writer_final.writerow({
            "cenario": scenario,
            "metodo": "BRKGA",
            "instancia": instance_name,
            "melhor custo": round(best_cost, 4),
            "gap melhor": round(gap_best, 2),
            "pior custo": round(worst_cost, 4),
            "dispersao": round(dispersion, 2),
        })

    # Grafico da melhor solução.

    plot_routes(
        [[0] + r + [0] for r in best_overall_routes],
        node_coord,
        d,
        optimal,
        gap_best,
        best_seed,
        instance_name,
        best_cost,
        K,
        os.path.join(output_dir, f"result_best_routes_{instance_name}.pdf"),
    )

    print(f"CSV salvo em:     {csv_path}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="BRKGA Cenário 1 — Porta-a-porta"
    )
    parser.add_argument("instance", type=str, help="Caminho para o .vrp")
    parser.add_argument("--config", type=str, default="configs/config.conf")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Ex: --seeds 1 2 3 4 5",
    )
    parser.add_argument("--max_time", type=int, default=60)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--scenario", type=str, default="C1")

    args = parser.parse_args()

    run_brkga(
        instance_path=args.instance,
        params_path=args.config,
        seeds=args.seeds,
        max_time=args.max_time,
        output_dir=args.output_dir,
        scenario=args.scenario,
    )
