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

SEED = 42
RANDOM_REPEATS = 100
BUDGET_LEVELS = [0.01, 0.05, 0.10]
STRATEGIES = {
    "random": None,
    "degree": "degree_centrality",
    "betweenness": "betweenness_centrality",
    "gnn": "gnn_infection_prob",
}

LOG_PATH = Path("logs/intervention.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)


def top_k_nodes(df: pd.DataFrame, score_col: str, k_nodes: int) -> list[str]:
    ranked = df.sort_values(
        by=[score_col, "weighted_degree_sec", "node_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ranked.head(k_nodes)["node_id"].tolist()


def coverage_for_edges(edges_df: pd.DataFrame, selected: set[str]) -> tuple[float, float]:
    incident = edges_df["source"].isin(selected) | edges_df["target"].isin(selected)
    total_duration = float(edges_df["weight"].sum())
    total_events = float(edges_df["n_contacts"].sum())
    duration_cov = (
        float(edges_df.loc[incident, "weight"].sum()) / total_duration * 100
        if total_duration > 0
        else 0.0
    )
    event_cov = (
        float(edges_df.loc[incident, "n_contacts"].sum()) / total_events * 100
        if total_events > 0
        else 0.0
    )
    return duration_cov, event_cov


def evaluate_selection(
    selected: list[str],
    all_scores: pd.DataFrame,
    edges_df: pd.DataFrame,
    trans_edges: pd.DataFrame,
    G: nx.Graph,
    lcc_before: int,
    n_positive: int,
) -> dict[str, float]:
    selected_set = set(selected)
    selected_df = all_scores[all_scores["node_id"].isin(selected_set)]
    positives_selected = int(selected_df["sars_label"].sum())
    precision_k = positives_selected / len(selected) * 100
    recall_k = positives_selected / n_positive * 100 if n_positive else 0.0

    covered_trans = trans_edges.apply(
        lambda row: row["source"] in selected_set or row["target"] in selected_set,
        axis=1,
    ).sum()
    transmission_coverage = covered_trans / len(trans_edges) * 100 if len(trans_edges) > 0 else 0.0
    duration_cov, event_cov = coverage_for_edges(edges_df, selected_set)

    G_intervened = G.copy()
    G_intervened.remove_nodes_from(selected)
    if G_intervened.number_of_nodes() > 0:
        lcc_after = max(len(c) for c in nx.connected_components(G_intervened))
    else:
        lcc_after = 0
    lcc_reduction = (lcc_before - lcc_after) / lcc_before * 100

    return {
        "precision_k_pct": precision_k,
        "recall_k_pct": recall_k,
        "transmission_coverage": transmission_coverage,
        "dynamic_duration_coverage_pct": duration_cov,
        "dynamic_contact_event_coverage_pct": event_cov,
        "lcc_reduction_pct": lcc_reduction,
    }


def main() -> None:
    logging.info("Budget allocation start")

    with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
        G = pickle.load(f)
    PATHS.intervention.mkdir(parents=True, exist_ok=True)

    scores_df = pd.read_csv(PATHS.metrics / "node_scores.csv")
    gnn_df = pd.read_csv(PATHS.model_results / "gnn_risk_scores.csv")
    edges_df = pd.read_csv(require_existing(PATHS.processed_edgelist, "processed SASHTS edgelist"))

    all_scores = scores_df.merge(
        gnn_df[["node_id", "gnn_infection_prob", "gnn_pred_sars"]],
        on="node_id",
        how="left",
    )
    all_scores["gnn_infection_prob"] = all_scores["gnn_infection_prob"].fillna(0.5)
    all_scores["gnn_pred_sars"] = all_scores["gnn_pred_sars"].fillna(0).astype(int)

    nodes = all_scores["node_id"].tolist()
    n_nodes = len(nodes)
    n_positive = int(all_scores["sars_label"].sum())
    lcc_before = max(len(c) for c in nx.connected_components(G))
    trans_edges = edges_df[edges_df["transmission"] == "Transmission"][["source", "target"]]

    rng = np.random.default_rng(SEED)
    results = []
    selected_map = {}
    random_replicates = {}

    print("=== BUDGET-CONSTRAINED TOP-K NODE SELECTION ===")
    print(f"Nodes: {n_nodes} | SARS+ baseline: {n_positive / n_nodes * 100:.1f}%")
    print(f"Transmission edges: {len(trans_edges)}")

    for k_pct in BUDGET_LEVELS:
        k_nodes = max(1, int(k_pct * n_nodes))
        budget_key = int(k_pct * 100)

        for strategy, score_col in STRATEGIES.items():
            if strategy == "random":
                replicate_rows = []
                replicate_selections = []
                for _ in range(RANDOM_REPEATS):
                    selected = rng.choice(nodes, size=k_nodes, replace=False).tolist()
                    replicate_selections.append(selected)
                    replicate_rows.append(
                        evaluate_selection(
                            selected=selected,
                            all_scores=all_scores,
                            edges_df=edges_df,
                            trans_edges=trans_edges,
                            G=G,
                            lcc_before=lcc_before,
                            n_positive=n_positive,
                        )
                    )
                metrics = pd.DataFrame(replicate_rows).mean().to_dict()
                selected = replicate_selections[0]
                random_replicates[f"random_k{budget_key}"] = replicate_selections
            else:
                selected = top_k_nodes(all_scores, score_col, k_nodes)
                metrics = evaluate_selection(
                    selected=selected,
                    all_scores=all_scores,
                    edges_df=edges_df,
                    trans_edges=trans_edges,
                    G=G,
                    lcc_before=lcc_before,
                    n_positive=n_positive,
                )
            selected_map[f"{strategy}_k{budget_key}"] = selected

            row = {
                "budget_k_pct": budget_key,
                "budget_k_nodes": k_nodes,
                "strategy": strategy,
                "precision_k_pct": round(metrics["precision_k_pct"], 1),
                "recall_k_pct": round(metrics["recall_k_pct"], 1),
                "transmission_coverage": round(metrics["transmission_coverage"], 1),
                "dynamic_duration_coverage_pct": round(
                    metrics["dynamic_duration_coverage_pct"], 1
                ),
                "dynamic_contact_event_coverage_pct": round(
                    metrics["dynamic_contact_event_coverage_pct"], 1
                ),
                "lcc_reduction_pct": round(metrics["lcc_reduction_pct"], 1),
            }
            results.append(row)

            print(
                f"[k={budget_key:2d}% | {strategy:>11}] "
                f"Prec@k={metrics['precision_k_pct']:5.1f}% | "
                f"Rec@k={metrics['recall_k_pct']:5.1f}% | "
                f"Trans.cover={metrics['transmission_coverage']:5.1f}% | "
                f"Dur.cover={metrics['dynamic_duration_coverage_pct']:5.1f}% | "
                f"LCC↓={metrics['lcc_reduction_pct']:5.1f}%"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(PATHS.intervention / "topk_budget_results.csv", index=False)

    with open(PATHS.intervention / "selected_nodes_by_strategy.json", "w", encoding="utf-8") as f:
        json.dump(selected_map, f, indent=2)

    with open(PATHS.intervention / "random_replicates_by_budget.json", "w", encoding="utf-8") as f:
        json.dump(random_replicates, f, indent=2)

    k1 = results_df[results_df["budget_k_pct"] == 1].sort_values(
        "transmission_coverage", ascending=False
    )
    best_k1 = k1.iloc[0]

    try:
        with open(PATHS.model_results / "gnn_metrics.json", encoding="utf-8") as f:
            gnn_metrics = json.load(f)
        gnn_auc = gnn_metrics["test"]["auc"]
    except (FileNotFoundError, KeyError):
        gnn_auc = None

    summary = {
        "n_nodes": n_nodes,
        "n_positive": n_positive,
        "budget_levels_pct": [int(v * 100) for v in BUDGET_LEVELS],
        "strategies": list(STRATEGIES.keys()),
        "random_repeats": RANDOM_REPEATS,
        "best_strategy_k1_by_transmission_coverage": {
            "strategy": best_k1["strategy"],
            "transmission_coverage": float(best_k1["transmission_coverage"]),
        },
        "gnn_test_auc": gnn_auc,
    }
    with open(PATHS.intervention / "topk_budget_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY: Transmission Coverage by budget ===")
    print(
        results_df.pivot_table(
            index="strategy",
            columns="budget_k_pct",
            values="transmission_coverage",
            aggfunc="first",
        ).to_string()
    )
    auc_text = f"{gnn_auc:.3f}" if gnn_auc is not None else "NA"
    print(
        f"\nBudget allocation completed. GNN AUC: {auc_text}. "
        f"Best strategy at k=1%: {best_k1['strategy']} "
        f"(trans.coverage={best_k1['transmission_coverage']:.1f}%)"
    )

    logging.info(
        "Budget allocation done | best_k1=%s trans_cov=%.1f",
        best_k1["strategy"],
        best_k1["transmission_coverage"],
    )


if __name__ == "__main__":
    main()
