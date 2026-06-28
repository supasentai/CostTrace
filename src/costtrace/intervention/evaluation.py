import json
import logging
import sys
from pathlib import Path

import pandas as pd

from costtrace.config import PATHS


sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = Path("logs/intervention.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")],
)


def main() -> None:
    logging.info("Intervention metrics summary start")

    PATHS.intervention.mkdir(parents=True, exist_ok=True)

    cf_df = pd.read_csv(PATHS.intervention / "counterfactual_results.csv")
    sir_df = pd.read_csv(PATHS.intervention / "sir_intervention_results.csv")
    topk_df = pd.read_csv(PATHS.intervention / "topk_budget_results.csv")

    final = cf_df.merge(
        sir_df[
            [
                "budget_k_pct",
                "strategy",
                "mean_infected_per_hh",
                "reduction_vs_baseline_pct",
            ]
        ],
        on=["budget_k_pct", "strategy"],
        how="inner",
    )
    final = final.merge(
        topk_df[
            [
                "budget_k_pct",
                "strategy",
                "precision_k_pct",
                "transmission_coverage",
                "dynamic_duration_coverage_pct",
                "dynamic_contact_event_coverage_pct",
            ]
        ],
        on=["budget_k_pct", "strategy"],
        how="inner",
    )

    final = final.sort_values(["budget_k_pct", "prevention_rate_pct"], ascending=[True, False])
    final.to_csv(PATHS.intervention / "final_comparison.csv", index=False)

    print("=" * 88)
    print("FINAL COMPARISON TABLE - SASHTS Budget-Constrained Intervention")
    print("=" * 88)
    print(
        f"{'Strategy':<13}{'Budget':>8}{'Prec@k':>9}{'TransCov':>10}"
        f"{'DurCov':>9}{'PrevRate':>10}{'SIR Red':>9}"
    )
    print("-" * 88)

    for _, row in final.iterrows():
        print(
            f"{row['strategy']:<13}{row['budget_k_pct']:>6.0f}%"
            f"{row['precision_k_pct']:>8.1f}%"
            f"{row['transmission_coverage']:>9.1f}%"
            f"{row['dynamic_duration_coverage_pct']:>8.1f}%"
            f"{row['prevention_rate_pct']:>9.1f}%"
            f"{row['reduction_vs_baseline_pct']:>8.1f}%"
        )

    best_by_budget = {}
    print("\n=== BEST STRATEGY PER BUDGET ===")
    for k_pct in sorted(final["budget_k_pct"].unique()):
        sub = final[final["budget_k_pct"] == k_pct].sort_values(
            ["prevention_rate_pct", "reduction_vs_baseline_pct", "transmission_coverage"],
            ascending=False,
        )
        best = sub.iloc[0]
        best_by_budget[str(int(k_pct))] = {
            "strategy": best["strategy"],
            "prevention_rate_pct": float(best["prevention_rate_pct"]),
            "transmission_coverage": float(best["transmission_coverage"]),
            "sir_reduction_pct": float(best["reduction_vs_baseline_pct"]),
        }
        print(
            f"k={int(k_pct):2d}%: {best['strategy'].upper():<11} | "
            f"Prevented={best['prevention_rate_pct']:.1f}% | "
            f"Trans.cover={best['transmission_coverage']:.1f}% | "
            f"SIR reduction={best['reduction_vs_baseline_pct']:.1f}%"
        )

    gnn_vs_random_k1 = None
    k1 = final[final["budget_k_pct"] == 1].set_index("strategy")
    if {"gnn", "random"}.issubset(k1.index):
        gnn_vs_random_k1 = float(
            k1.loc["gnn", "prevention_rate_pct"] - k1.loc["random", "prevention_rate_pct"]
        )

    with open(PATHS.intervention / "final_strategy_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_by_budget": best_by_budget,
                "gnn_minus_random_prevention_rate_k1_pct": gnn_vs_random_k1,
                "n_rows": int(len(final)),
            },
            f,
            indent=2,
        )

    print("\nIntervention evaluation completed.")
    logging.info("Intervention metrics summary done | rows=%s", len(final))


if __name__ == "__main__":
    main()
