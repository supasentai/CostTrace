from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from costtrace.config import PATHS, phase05_config, require_existing
from costtrace.intervention.allocation import evaluate_selection, top_k_nodes
from costtrace.intervention.counterfactual import evaluate_selected_set
from costtrace.intervention.simulation import household_components, infection_probability


SCORE_COLUMNS = {
    "degree": "degree_centrality",
    "betweenness": "betweenness_centrality",
    "gnn": "gnn_infection_prob",
}


def load_inputs() -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
        graph = pickle.load(f)
    meta = pd.read_csv(require_existing(PATHS.processed_metadata, "processed SASHTS metadata"))
    edges = pd.read_csv(require_existing(PATHS.processed_edgelist, "processed SASHTS edgelist"))
    scores = pd.read_csv(PATHS.metrics / "node_scores.csv")
    gnn = pd.read_csv(PATHS.model_results / "gnn_risk_scores.csv")

    for frame, columns in [
        (meta, ["indid", "hhid"]),
        (edges, ["source", "target"]),
        (scores, ["node_id"]),
        (gnn, ["node_id"]),
    ]:
        for column in columns:
            frame[column] = frame[column].astype(str)

    all_scores = scores.merge(
        gnn[["node_id", "gnn_infection_prob", "gnn_pred_sars"]],
        on="node_id",
        how="left",
    )
    all_scores["gnn_infection_prob"] = all_scores["gnn_infection_prob"].fillna(0.5)
    all_scores["gnn_pred_sars"] = all_scores["gnn_pred_sars"].fillna(0).astype(int)
    return graph, meta, edges, all_scores


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def make_param_id(beta: float, gamma: float) -> str:
    return f"beta{beta:.2f}_gamma{gamma:.2f}"


def scenario_seed(base_seed: int, param_idx: int, run_id: int, budget_pct: int, strategy_idx: int) -> int:
    return base_seed + param_idx * 1_000_000 + run_id * 10_000 + budget_pct * 100 + strategy_idx


def select_nodes(
    strategy: str,
    budget_pct: int,
    run_id: int,
    seed: int,
    nodes: list[str],
    all_scores: pd.DataFrame,
) -> list[str]:
    k_nodes = max(1, int((budget_pct / 100.0) * len(nodes)))
    if strategy == "random":
        rng = np.random.default_rng(seed + run_id)
        return rng.choice(nodes, size=k_nodes, replace=False).tolist()
    return top_k_nodes(all_scores, SCORE_COLUMNS[strategy], k_nodes)


def sir_timeseries_on_household(
    sg: nx.Graph,
    index_node: str | None,
    max_duration_sec: float,
    beta: float,
    gamma: float,
    t_max: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    state = {node: "S" for node in sorted(sg.nodes())}
    total_ever_infected = 0
    if index_node in state:
        state[index_node] = "I"
        total_ever_infected = 1

    series = [state_counts(state, total_ever_infected, 0)]
    for time_step in range(1, t_max + 1):
        newly_infected: set[str] = set()
        newly_recovered: set[str] = set()

        for node in sorted(state):
            if state[node] != "I":
                continue
            for neighbor in sorted(sg.neighbors(node)):
                if state[neighbor] != "S" or neighbor in newly_infected:
                    continue
                duration = float(sg[node][neighbor].get("total_duration_sec", 60.0))
                if rng.random() < infection_probability(duration, max_duration_sec, beta):
                    newly_infected.add(neighbor)
            if rng.random() < gamma:
                newly_recovered.add(node)

        for node in newly_infected:
            state[node] = "I"
        for node in newly_recovered:
            if state[node] == "I":
                state[node] = "R"

        total_ever_infected += len(newly_infected)
        series.append(state_counts(state, total_ever_infected, time_step))

    return series


def state_counts(state: dict[str, str], total_ever_infected: int, time_step: int) -> dict[str, int]:
    values = list(state.values())
    return {
        "time_step": time_step,
        "susceptible": values.count("S"),
        "infected_active": values.count("I"),
        "recovered": values.count("R"),
        "ever_infected": total_ever_infected,
    }


def simulate_scenario(
    comps: dict[str, nx.Graph],
    index_by_hh: dict[str, str],
    selected_nodes: set[str],
    max_duration_sec: float,
    beta: float,
    gamma: float,
    t_max: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    household_series: list[list[dict[str, int]]] = []
    final_counts: list[int] = []

    for hh_pos, hhid in enumerate(sorted(comps)):
        sg = comps[hhid].copy()
        if selected_nodes:
            sg.remove_nodes_from([node for node in selected_nodes if node in sg])
        rng = np.random.default_rng(seed + hh_pos)
        series = sir_timeseries_on_household(
            sg=sg,
            index_node=index_by_hh.get(hhid),
            max_duration_sec=max_duration_sec,
            beta=beta,
            gamma=gamma,
            t_max=t_max,
            rng=rng,
        )
        household_series.append(series)
        final_counts.append(series[-1]["ever_infected"])

    n_households = len(household_series)
    aggregated = []
    for time_step in range(t_max + 1):
        row = {"time_step": time_step}
        for field in ["susceptible", "infected_active", "recovered", "ever_infected"]:
            row[f"mean_{field}"] = float(
                np.mean([series[time_step][field] for series in household_series])
            )
        row["n_households"] = n_households
        aggregated.append(row)

    return float(np.mean(final_counts)), aggregated


def summarize_baseline(baseline_rows: list[dict[str, float]]) -> pd.DataFrame:
    baseline_df = pd.DataFrame(baseline_rows)
    rows = []
    for (param_id, beta, gamma, profile), group in baseline_df.groupby(
        ["param_id", "beta", "gamma", "profile"], sort=True
    ):
        low, high = percentile_interval(group["final_mean_infected_per_hh"])
        rows.append(
            {
                "profile": profile,
                "param_id": param_id,
                "beta": beta,
                "gamma": gamma,
                "n_runs": int(len(group)),
                "mean_baseline_infected_per_hh": round(
                    float(group["final_mean_infected_per_hh"].mean()), 4
                ),
                "std_baseline_infected_per_hh": round(
                    float(group["final_mean_infected_per_hh"].std(ddof=0)), 4
                ),
                "p2_5_baseline_infected_per_hh": round(low, 4),
                "p97_5_baseline_infected_per_hh": round(high, 4),
            }
        )
    return pd.DataFrame(rows)


def summarize_parameter_sweep(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = run_df[run_df["strategy"] != "baseline"].copy()
    for keys, group in work.groupby(
        ["profile", "param_id", "beta", "gamma", "strategy", "budget_k_pct", "budget_k_nodes"],
        sort=True,
    ):
        profile, param_id, beta, gamma, strategy, budget_pct, budget_nodes = keys
        red_low, red_high = percentile_interval(group["reduction_vs_baseline_pct"])
        inf_low, inf_high = percentile_interval(group["final_mean_infected_per_hh"])
        rows.append(
            {
                "profile": profile,
                "param_id": param_id,
                "beta": beta,
                "gamma": gamma,
                "strategy": strategy,
                "budget_k_pct": int(budget_pct),
                "budget_k_nodes": int(budget_nodes),
                "mean_infected_per_hh": round(float(group["final_mean_infected_per_hh"].mean()), 4),
                "std_infected_per_hh": round(float(group["final_mean_infected_per_hh"].std(ddof=0)), 4),
                "p2_5_infected_per_hh": round(inf_low, 4),
                "p97_5_infected_per_hh": round(inf_high, 4),
                "mean_reduction_vs_baseline_pct": round(
                    float(group["reduction_vs_baseline_pct"].mean()), 4
                ),
                "std_reduction_vs_baseline_pct": round(
                    float(group["reduction_vs_baseline_pct"].std(ddof=0)), 4
                ),
                "p2_5_reduction_vs_baseline_pct": round(red_low, 4),
                "p97_5_reduction_vs_baseline_pct": round(red_high, 4),
                "n_runs": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_budget_curve(run_df: pd.DataFrame, default_beta: float, default_gamma: float) -> pd.DataFrame:
    work = run_df[
        (run_df["strategy"] != "baseline")
        & np.isclose(run_df["beta"], default_beta)
        & np.isclose(run_df["gamma"], default_gamma)
    ].copy()
    agg = (
        work.groupby(["profile", "beta", "gamma", "strategy", "budget_k_pct", "budget_k_nodes"], sort=True)
        .agg(
            mean_infected_per_hh=("final_mean_infected_per_hh", "mean"),
            reduction_vs_baseline_pct=("reduction_vs_baseline_pct", "mean"),
            prevention_rate_pct=("prevention_rate_pct", "mean"),
            transmission_coverage=("transmission_coverage", "mean"),
            precision_k_pct=("precision_k_pct", "mean"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
        .sort_values(["strategy", "budget_k_pct"])
    )
    agg["marginal_sir_reduction_gain_pct"] = (
        agg.groupby("strategy")["reduction_vs_baseline_pct"].diff().fillna(agg["reduction_vs_baseline_pct"])
    )
    return agg.round(4)


def summarize_decision_table(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget_pct, group in curve_df.groupby("budget_k_pct", sort=True):
        best_prevention = group.sort_values("prevention_rate_pct", ascending=False).iloc[0]
        best_sir = group.sort_values("reduction_vs_baseline_pct", ascending=False).iloc[0]
        best_coverage = group.sort_values("transmission_coverage", ascending=False).iloc[0]
        rows.append(
            {
                "budget_k_pct": int(budget_pct),
                "budget_k_nodes": int(best_prevention["budget_k_nodes"]),
                "best_strategy_by_prevention": best_prevention["strategy"],
                "best_prevention_rate_pct": round(float(best_prevention["prevention_rate_pct"]), 4),
                "best_strategy_by_sir_reduction": best_sir["strategy"],
                "best_sir_reduction_pct": round(float(best_sir["reduction_vs_baseline_pct"]), 4),
                "best_strategy_by_transmission_coverage": best_coverage["strategy"],
                "best_transmission_coverage_pct": round(float(best_coverage["transmission_coverage"]), 4),
            }
        )
    return pd.DataFrame(rows)


def summarize_uncertainty(run_df: pd.DataFrame, default_beta: float, default_gamma: float) -> pd.DataFrame:
    work = run_df[
        (run_df["strategy"] != "baseline")
        & np.isclose(run_df["beta"], default_beta)
        & np.isclose(run_df["gamma"], default_gamma)
    ].copy()
    metric_map = {
        "final_mean_infected_per_hh": "final_mean_infected_per_hh",
        "reduction_vs_baseline_pct": "reduction_vs_baseline_pct",
        "prevention_rate_pct": "prevention_rate_pct",
    }
    rows = []
    for keys, group in work.groupby(["profile", "strategy", "budget_k_pct", "budget_k_nodes"], sort=True):
        profile, strategy, budget_pct, budget_nodes = keys
        for metric_name, column in metric_map.items():
            low, high = percentile_interval(group[column])
            rows.append(
                {
                    "profile": profile,
                    "strategy": strategy,
                    "budget_k_pct": int(budget_pct),
                    "budget_k_nodes": int(budget_nodes),
                    "metric_name": metric_name,
                    "point_estimate": round(float(group[column].mean()), 4),
                    "lower_interval": round(low, 4),
                    "upper_interval": round(high, 4),
                    "method": "percentile_interval_across_phase05_runs",
                    "n_resamples_or_runs": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def write_figures(
    timeseries_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
) -> None:
    PATHS.phase05_figures.mkdir(parents=True, exist_ok=True)

    default_ts = timeseries_df[
        (timeseries_df["param_id"] == "beta0.25_gamma0.10")
        & (timeseries_df["budget_k_pct"].isin([0, 10]))
    ].copy()
    ts_plot = (
        default_ts.groupby(["strategy", "time_step"], sort=True)["mean_ever_infected"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy, group in ts_plot.groupby("strategy", sort=True):
        ax.plot(group["time_step"], group["mean_ever_infected"], marker="o", label=strategy)
    ax.set_title("Phase 05 smoke time-series curve")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean ever infected per household")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS.phase05_figures / "phase05_timeseries_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for strategy, group in curve_df.groupby("strategy", sort=True):
        ax.plot(
            group["budget_k_pct"],
            group["reduction_vs_baseline_pct"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Phase 05 budget curve")
    ax.set_xlabel("Budget (% of nodes)")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS.phase05_figures / "phase05_budget_curve.png", dpi=160)
    plt.close(fig)

    beta_plot = parameter_df[
        (parameter_df["budget_k_pct"] == 10) & (parameter_df["strategy"] == "gnn")
    ].sort_values("beta")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(beta_plot["beta"], beta_plot["mean_reduction_vs_baseline_pct"], marker="o")
    ax.set_title("Phase 05 parameter sweep: GNN at 10% budget")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS.phase05_figures / "phase05_parameter_sweep.png", dpi=160)
    plt.close(fig)

    unc_plot = uncertainty_df[
        (uncertainty_df["metric_name"] == "reduction_vs_baseline_pct")
        & (uncertainty_df["budget_k_pct"].isin([1, 5, 10, 20]))
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    offsets = {"random": -0.3, "degree": -0.1, "betweenness": 0.1, "gnn": 0.3}
    for strategy, group in unc_plot.groupby("strategy", sort=True):
        x = group["budget_k_pct"].astype(float) + offsets.get(strategy, 0)
        y = group["point_estimate"]
        yerr = np.vstack([y - group["lower_interval"], group["upper_interval"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label=strategy)
    ax.set_title("Phase 05 uncertainty intervals")
    ax.set_xlabel("Budget (% of nodes)")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS.phase05_figures / "phase05_uncertainty_intervals.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 05 scenario-evaluation artifacts.")
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    cfg = phase05_config(args.profile)
    PATHS.intervention.mkdir(parents=True, exist_ok=True)

    graph, meta, edges, all_scores = load_inputs()
    nodes = sorted(all_scores["node_id"].astype(str).tolist())
    comps = household_components(graph)
    index_by_hh = meta[meta["index"] == "Index"].set_index("hhid")["indid"].to_dict()
    max_duration_sec = max(float(attrs.get("total_duration_sec", 1.0)) for _, _, attrs in graph.edges(data=True))
    trans_edges = edges[edges["transmission"] == "Transmission"][["source", "target"]]
    index_nodes = set(meta[meta["index"] == "Index"]["indid"])
    contact_nodes = set(meta[meta["index"] == "Contact"]["indid"])
    sars_positive_contacts = set(
        meta[(meta["sars"] == "Positive") & (meta["index"] == "Contact")]["indid"]
    )
    n_positive = int(all_scores["sars_label"].sum())
    lcc_before = max(len(c) for c in nx.connected_components(graph))

    run_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    baseline_by_param_run: dict[tuple[str, int], float] = {}

    for param_idx, (beta, gamma) in enumerate(cfg.parameter_grid):
        param_id = make_param_id(beta, gamma)
        for run_id in range(cfg.n_runs):
            seed = scenario_seed(cfg.random_seed, param_idx, run_id, 0, 0)
            baseline_mean, baseline_series = simulate_scenario(
                comps=comps,
                index_by_hh=index_by_hh,
                selected_nodes=set(),
                max_duration_sec=max_duration_sec,
                beta=beta,
                gamma=gamma,
                t_max=cfg.t_max,
                seed=seed,
            )
            baseline_by_param_run[(param_id, run_id)] = baseline_mean
            baseline_row = {
                "profile": cfg.profile,
                "param_id": param_id,
                "beta": beta,
                "gamma": gamma,
                "run_id": run_id,
                "seed": seed,
                "budget_k_pct": 0,
                "budget_k_nodes": 0,
                "strategy": "baseline",
                "selected_node_ids": "",
                "final_mean_infected_per_hh": baseline_mean,
                "baseline_mean_infected_per_hh": baseline_mean,
                "reduction_vs_baseline_pct": 0.0,
                "transmissions_blocked": 0.0,
                "transmission_block_rate_pct": 0.0,
                "infections_prevented": 0.0,
                "prevention_rate_pct": 0.0,
                "precision_k_pct": 0.0,
                "transmission_coverage": 0.0,
            }
            run_rows.append(baseline_row)
            baseline_rows.append(baseline_row)
            for series_row in baseline_series:
                timeseries_rows.append(
                    {
                        "profile": cfg.profile,
                        "param_id": param_id,
                        "beta": beta,
                        "gamma": gamma,
                        "run_id": run_id,
                        "seed": seed,
                        "budget_k_pct": 0,
                        "budget_k_nodes": 0,
                        "strategy": "baseline",
                        **series_row,
                    }
                )

        for budget_pct in cfg.budget_levels_pct:
            for strategy_idx, strategy in enumerate(cfg.strategies, start=1):
                for run_id in range(cfg.n_runs):
                    seed = scenario_seed(cfg.random_seed, param_idx, run_id, budget_pct, strategy_idx)
                    selected = select_nodes(strategy, budget_pct, run_id, seed, nodes, all_scores)
                    selected_set = set(selected)
                    budget_nodes = len(selected)

                    topk_metrics = evaluate_selection(
                        selected=selected,
                        all_scores=all_scores,
                        edges_df=edges,
                        trans_edges=trans_edges,
                        G=graph,
                        lcc_before=lcc_before,
                        n_positive=n_positive,
                    )
                    cf_metrics = evaluate_selected_set(
                        selected=selected_set,
                        trans_df=edges[edges["transmission"] == "Transmission"],
                        total_transmissions=len(trans_edges),
                        index_nodes=index_nodes,
                        contact_nodes=contact_nodes,
                        sars_positive_contacts=sars_positive_contacts,
                    )
                    final_mean, series = simulate_scenario(
                        comps=comps,
                        index_by_hh=index_by_hh,
                        selected_nodes=selected_set,
                        max_duration_sec=max_duration_sec,
                        beta=beta,
                        gamma=gamma,
                        t_max=cfg.t_max,
                        seed=seed,
                    )
                    baseline_mean = baseline_by_param_run[(param_id, run_id)]
                    reduction = (
                        (baseline_mean - final_mean) / baseline_mean * 100
                        if baseline_mean > 0
                        else 0.0
                    )

                    run_rows.append(
                        {
                            "profile": cfg.profile,
                            "param_id": param_id,
                            "beta": beta,
                            "gamma": gamma,
                            "run_id": run_id,
                            "seed": seed,
                            "budget_k_pct": budget_pct,
                            "budget_k_nodes": budget_nodes,
                            "strategy": strategy,
                            "selected_node_ids": "|".join(selected),
                            "final_mean_infected_per_hh": final_mean,
                            "baseline_mean_infected_per_hh": baseline_mean,
                            "reduction_vs_baseline_pct": reduction,
                            **cf_metrics,
                            **topk_metrics,
                        }
                    )
                    for series_row in series:
                        timeseries_rows.append(
                            {
                                "profile": cfg.profile,
                                "param_id": param_id,
                                "beta": beta,
                                "gamma": gamma,
                                "run_id": run_id,
                                "seed": seed,
                                "budget_k_pct": budget_pct,
                                "budget_k_nodes": budget_nodes,
                                "strategy": strategy,
                                **series_row,
                            }
                        )

    run_df = pd.DataFrame(run_rows).round(4)
    timeseries_df = pd.DataFrame(timeseries_rows).round(4)
    baseline_df = summarize_baseline(baseline_rows)
    parameter_df = summarize_parameter_sweep(run_df)
    curve_df = summarize_budget_curve(run_df, cfg.default_beta, cfg.default_gamma)
    decision_df = summarize_decision_table(curve_df)
    uncertainty_df = summarize_uncertainty(run_df, cfg.default_beta, cfg.default_gamma)

    run_df.to_csv(PATHS.intervention / cfg.run_level_results, index=False)
    timeseries_df.to_csv(PATHS.intervention / cfg.timeseries_results, index=False)
    baseline_df.to_csv(PATHS.intervention / cfg.baseline_summary, index=False)
    parameter_df.to_csv(PATHS.intervention / cfg.parameter_sweep, index=False)
    curve_df.to_csv(PATHS.intervention / cfg.budget_curve, index=False)
    decision_df.to_csv(PATHS.intervention / cfg.budget_decision_table, index=False)
    uncertainty_df.to_csv(PATHS.intervention / cfg.uncertainty_summary, index=False)

    metadata = {
        "profile": cfg.profile,
        "random_seed": cfg.random_seed,
        "n_runs": cfg.n_runs,
        "t_max": cfg.t_max,
        "beta_values": list(cfg.beta_values),
        "gamma_values": list(cfg.gamma_values),
        "default_beta": cfg.default_beta,
        "default_gamma": cfg.default_gamma,
        "budget_levels_pct": list(cfg.budget_levels_pct),
        "strategies": list(cfg.strategies),
        "note": "Phase 05 artifacts are generated from saved inputs and existing intervention logic; no model training is performed.",
    }
    with open(PATHS.intervention / "phase05_generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.skip_figures:
        write_figures(timeseries_df, parameter_df, curve_df, uncertainty_df)

    print(f"Generated Phase 05 artifacts with profile={cfg.profile}")
    print(f"Run-level rows: {len(run_df)}")
    print(f"Time-series rows: {len(timeseries_df)}")


if __name__ == "__main__":
    main()
