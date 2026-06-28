import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

from costtrace.config import PATHS, require_existing


sys.stdout.reconfigure(encoding="utf-8")

BUDGET_LEVELS = [1, 5, 10]
STRATEGIES = ["random", "degree", "betweenness", "gnn"]

LOG_PATH = Path("logs/intervention.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)


def prevented_secondary_cases(
    blocked_transmissions: pd.DataFrame,
    sars_positive_contacts: set[str],
) -> set[str]:
    prevented = set()
    for _, row in blocked_transmissions.iterrows():
        for node in [row["source"], row["target"]]:
            if node in sars_positive_contacts:
                prevented.add(node)
    return prevented


def evaluate_selected_set(
    selected: set[str],
    trans_df: pd.DataFrame,
    total_transmissions: int,
    index_nodes: set[str],
    contact_nodes: set[str],
    sars_positive_contacts: set[str],
) -> dict[str, float]:
    blocked = trans_df[
        trans_df["source"].isin(selected) | trans_df["target"].isin(selected)
    ]
    prevented_cases = prevented_secondary_cases(blocked, sars_positive_contacts)

    n_blocked = len(blocked)
    n_prevented = len(prevented_cases)
    transmission_block_rate = n_blocked / total_transmissions * 100 if total_transmissions else 0.0
    prevention_rate = (
        n_prevented / len(sars_positive_contacts) * 100 if sars_positive_contacts else 0.0
    )

    selected_index = selected & index_nodes
    selected_contacts = selected & contact_nodes
    selected_sars_contacts = selected & sars_positive_contacts

    return {
        "transmissions_blocked": float(n_blocked),
        "transmission_block_rate_pct": transmission_block_rate,
        "infections_prevented": float(n_prevented),
        "prevention_rate_pct": prevention_rate,
        "selected_index_cases": float(len(selected_index)),
        "selected_contacts": float(len(selected_contacts)),
        "sars_pos_contacts_in_selected": float(len(selected_sars_contacts)),
    }


def main() -> None:
    logging.info("Counterfactual analysis start")

    with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
        _ = pickle.load(f)
    edges_df = pd.read_csv(require_existing(PATHS.processed_edgelist, "processed SASHTS edgelist"))
    meta = pd.read_csv(require_existing(PATHS.processed_metadata, "processed SASHTS metadata"))

    PATHS.intervention.mkdir(parents=True, exist_ok=True)

    with open(PATHS.intervention / "selected_nodes_by_strategy.json", encoding="utf-8") as f:
        selected_map = json.load(f)
    random_replicates = {}
    random_replicates_path = PATHS.intervention / "random_replicates_by_budget.json"
    if random_replicates_path.exists():
        with open(random_replicates_path, encoding="utf-8") as f:
            random_replicates = json.load(f)

    trans_df = edges_df[edges_df["transmission"] == "Transmission"].copy()
    total_transmissions = len(trans_df)

    index_nodes = set(meta[meta["index"] == "Index"]["indid"])
    contact_nodes = set(meta[meta["index"] == "Contact"]["indid"])
    sars_positive_contacts = set(
        meta[(meta["sars"] == "Positive") & (meta["index"] == "Contact")]["indid"]
    )

    print("=== COUNTERFACTUAL TRANSMISSION ANALYSIS ===")
    print(f"Total transmission edges       : {total_transmissions}")
    print(f"SARS+ secondary/contact cases  : {len(sars_positive_contacts)}")

    rows = []
    for k_pct in BUDGET_LEVELS:
        for strategy in STRATEGIES:
            key = f"{strategy}_k{k_pct}"
            if strategy == "random" and key in random_replicates:
                selection_sets = [set(nodes) for nodes in random_replicates[key]]
            else:
                selection_sets = [set(selected_map.get(key, []))]

            metrics = pd.DataFrame(
                [
                    evaluate_selected_set(
                        selected=selected,
                        trans_df=trans_df,
                        total_transmissions=total_transmissions,
                        index_nodes=index_nodes,
                        contact_nodes=contact_nodes,
                        sars_positive_contacts=sars_positive_contacts,
                    )
                    for selected in selection_sets
                ]
            ).mean()

            row = {
                "budget_k_pct": k_pct,
                "budget_k_nodes": len(selection_sets[0]) if selection_sets else 0,
                "strategy": strategy,
                "transmissions_blocked": round(float(metrics["transmissions_blocked"]), 2),
                "transmission_block_rate_pct": round(
                    float(metrics["transmission_block_rate_pct"]), 1
                ),
                "infections_prevented": round(float(metrics["infections_prevented"]), 2),
                "prevention_rate_pct": round(float(metrics["prevention_rate_pct"]), 1),
                "selected_index_cases": round(float(metrics["selected_index_cases"]), 2),
                "selected_contacts": round(float(metrics["selected_contacts"]), 2),
                "sars_pos_contacts_in_selected": round(
                    float(metrics["sars_pos_contacts_in_selected"]), 2
                ),
            }
            rows.append(row)

            print(
                f"[k={k_pct:2d}% | {strategy:>11}] "
                f"Trans.blocked={metrics['transmissions_blocked']:5.1f}/{total_transmissions} "
                f"({metrics['transmission_block_rate_pct']:5.1f}%) | "
                f"Secondary prevented={metrics['infections_prevented']:5.1f} "
                f"({metrics['prevention_rate_pct']:5.1f}%) | "
                f"Index selected={metrics['selected_index_cases']:4.1f}"
            )

    cf_df = pd.DataFrame(rows)
    cf_df.to_csv(PATHS.intervention / "counterfactual_results.csv", index=False)

    print("\n=== SUMMARY: Secondary Infections Prevented (%) ===")
    print(
        cf_df.pivot_table(
            index="strategy",
            columns="budget_k_pct",
            values="prevention_rate_pct",
            aggfunc="first",
        ).to_string()
    )

    logging.info("Counterfactual analysis done")


if __name__ == "__main__":
    main()
