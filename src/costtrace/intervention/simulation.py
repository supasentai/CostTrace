import json
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from costtrace.config import PATHS, require_existing


sys.stdout.reconfigure(encoding="utf-8")

BETA = 0.25
GAMMA = 0.10
T_MAX = 30
N_RUNS = 50
SEED = 42
MAX_RANDOM_REPEATS_FOR_SIR = 20
BUDGET_LEVELS = [1, 5, 10]
STRATEGIES = ["random", "degree", "betweenness", "gnn"]

LOG_PATH = Path("logs/intervention.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)


def infection_probability(duration_sec: float, max_duration_sec: float, beta: float) -> float:
    if max_duration_sec <= 0:
        return beta
    duration_factor = np.log1p(max(duration_sec, 0.0)) / np.log1p(max_duration_sec)
    return float(min(beta * (0.25 + 0.75 * duration_factor), 0.90))


def sir_on_household(
    sg: nx.Graph,
    index_node: str,
    max_duration_sec: float,
    beta: float,
    gamma: float,
    t_max: int,
    rng: np.random.Generator,
) -> int:
    if index_node not in sg:
        return 0

    state = {node: "S" for node in sorted(sg.nodes())}
    state[index_node] = "I"
    total_ever_infected = 1

    for _ in range(t_max):
        newly_infected = set()
        newly_recovered = set()

        for node in sorted(state):
            node_state = state[node]
            if node_state != "I":
                continue

            for neighbor in sorted(sg.neighbors(node)):
                if state[neighbor] != "S" or neighbor in newly_infected:
                    continue
                duration = float(sg[node][neighbor].get("total_duration_sec", 60.0))
                prob = infection_probability(duration, max_duration_sec, beta)
                if rng.random() < prob:
                    newly_infected.add(neighbor)

            if rng.random() < gamma:
                newly_recovered.add(node)

        for node in newly_infected:
            state[node] = "I"
        for node in newly_recovered:
            if state[node] == "I":
                state[node] = "R"

        total_ever_infected += len(newly_infected)
        if not any(node_state == "I" for node_state in state.values()):
            break

    return total_ever_infected


def household_components(G: nx.Graph) -> dict[str, nx.Graph]:
    comps = {}
    for comp in sorted(nx.connected_components(G), key=lambda nodes: min(nodes)):
        nodes = sorted(comp)
        sg = nx.Graph()
        for node in nodes:
            sg.add_node(node, **G.nodes[node])
        for u, v, attrs in sorted(G.subgraph(nodes).edges(data=True)):
            sg.add_edge(u, v, **attrs)
        first_node = nodes[0]
        hhid = sg.nodes[first_node].get("hhid")
        if hhid:
            comps[hhid] = sg
    return comps


def run_all_households(
    comps: dict[str, nx.Graph],
    index_by_hh: dict[str, str],
    selected_nodes: set[str],
    max_duration_sec: float,
    seed_offset: int,
) -> list[int]:
    infected_counts = []

    for run in range(N_RUNS):
        for hh_pos, hhid in enumerate(sorted(comps)):
            sg = comps[hhid].copy()
            if selected_nodes:
                sg.remove_nodes_from([node for node in selected_nodes if node in sg])
            index_node = index_by_hh.get(hhid)
            rng = np.random.default_rng(SEED + seed_offset + run * 1009 + hh_pos)
            infected_counts.append(
                sir_on_household(
                    sg=sg,
                    index_node=index_node,
                    max_duration_sec=max_duration_sec,
                    beta=BETA,
                    gamma=GAMMA,
                    t_max=T_MAX,
                    rng=rng,
                )
            )

    return infected_counts


def summarize_counts(counts: list[int]) -> dict[str, float]:
    arr = np.array(counts, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


def main() -> None:
    logging.info("SIR simulation start")

    with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
        G = pickle.load(f)
    meta = pd.read_csv(require_existing(PATHS.processed_metadata, "processed SASHTS metadata"))

    PATHS.intervention.mkdir(parents=True, exist_ok=True)

    with open(PATHS.intervention / "selected_nodes_by_strategy.json", encoding="utf-8") as f:
        selected_map = json.load(f)
    random_replicates = {}
    random_replicates_path = PATHS.intervention / "random_replicates_by_budget.json"
    if random_replicates_path.exists():
        with open(random_replicates_path, encoding="utf-8") as f:
            random_replicates = json.load(f)

    comps = household_components(G)
    index_by_hh = meta[meta["index"] == "Index"].set_index("hhid")["indid"].to_dict()
    max_duration_sec = max(
        float(attrs.get("total_duration_sec", 1.0)) for _, _, attrs in G.edges(data=True)
    )
    actual_attack_rate = (meta["sars"] == "Positive").mean() * 100

    print("=== SIR INTERVENTION SIMULATION ===")
    print(
        f"beta={BETA:.2f} | gamma={GAMMA:.2f} | T={T_MAX} | runs={N_RUNS} | "
        f"households={len(comps)}"
    )
    print(f"Observed SARS attack rate: {actual_attack_rate:.1f}%")

    baseline_counts = run_all_households(
        comps=comps,
        index_by_hh=index_by_hh,
        selected_nodes=set(),
        max_duration_sec=max_duration_sec,
        seed_offset=0,
    )
    baseline = summarize_counts(baseline_counts)
    print(
        f"Baseline mean infected per HH: {baseline['mean']:.2f} ± {baseline['std']:.2f} "
        f"(95% CI {baseline['ci95_low']:.1f}-{baseline['ci95_high']:.1f})"
    )

    rows = []
    for k_pct in BUDGET_LEVELS:
        for strategy in STRATEGIES:
            key = f"{strategy}_k{k_pct}"
            if strategy == "random" and key in random_replicates:
                selection_sets = [
                    set(nodes) for nodes in random_replicates[key][:MAX_RANDOM_REPEATS_FOR_SIR]
                ]
            else:
                selection_sets = [set(selected_map.get(key, []))]

            counts = []
            for rep_idx, selected in enumerate(selection_sets):
                counts.extend(
                    run_all_households(
                        comps=comps,
                        index_by_hh=index_by_hh,
                        selected_nodes=selected,
                        max_duration_sec=max_duration_sec,
                        seed_offset=(
                            k_pct * 10_000
                            + STRATEGIES.index(strategy) * 1_000
                            + rep_idx * 100_000
                        ),
                    )
                )
            stats = summarize_counts(counts)
            reduction = (
                (baseline["mean"] - stats["mean"]) / baseline["mean"] * 100
                if baseline["mean"] > 0
                else 0.0
            )

            row = {
                "budget_k_pct": k_pct,
                "strategy": strategy,
                "mean_infected_per_hh": round(stats["mean"], 2),
                "std_infected_per_hh": round(stats["std"], 2),
                "ci95_low_infected_per_hh": round(stats["ci95_low"], 2),
                "ci95_high_infected_per_hh": round(stats["ci95_high"], 2),
                "baseline_mean_infected": round(baseline["mean"], 2),
                "reduction_vs_baseline_pct": round(reduction, 1),
            }
            rows.append(row)

            print(
                f"[k={k_pct:2d}% | {strategy:>11}] "
                f"Mean infected={stats['mean']:.2f} ± {stats['std']:.2f} "
                f"(↓{reduction:5.1f}% vs baseline)"
            )

    sir_df = pd.DataFrame(rows)
    sir_df.to_csv(PATHS.intervention / "sir_intervention_results.csv", index=False)

    with open(PATHS.intervention / "sir_baseline.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "beta": BETA,
                "gamma": GAMMA,
                "t_max": T_MAX,
                "n_runs": N_RUNS,
                "baseline_mean_infected_per_hh": round(baseline["mean"], 4),
                "baseline_std_infected_per_hh": round(baseline["std"], 4),
                "baseline_ci95_low": round(baseline["ci95_low"], 4),
                "baseline_ci95_high": round(baseline["ci95_high"], 4),
                "observed_attack_rate_pct": round(actual_attack_rate, 1),
                "duration_scaling": "log1p(total_duration_sec) / log1p(max_duration_sec)",
                "note": "SIR runs are performed per household component with index case as seed.",
            },
            f,
            indent=2,
        )

    logging.info("SIR simulation done | baseline_mean=%.4f", baseline["mean"])


if __name__ == "__main__":
    main()
