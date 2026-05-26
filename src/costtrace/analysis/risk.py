import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log_path = Path("logs/analysis.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
)

logging.info("Risk score synthesis start")

centrality_df = pd.read_csv("results/centrality_scores.csv").sort_values("node_id")

# Encode categorical features for future GNN use
centrality_df["sex_enc"] = (centrality_df["sex"] == "Male").astype(int)
centrality_df["site_enc"] = (centrality_df["site"] == "Soweto").astype(int)
centrality_df["sleep_room_enc"] = (centrality_df["sleep_room_ix"] == "Yes").astype(int)
centrality_df["cared_by_enc"] = (centrality_df["cared_by_ix"] == "Yes").astype(int)
centrality_df["sus_enc"] = (centrality_df["sus"] == "Susceptible").astype(int)
centrality_df["sars_label"] = (centrality_df["sars"] == "Positive").astype(int)

# Age group encode (ordinal)
age_map = {"<5": 0, "5-12": 1, "13-17": 2, "18-34": 3, "35-59": 4, "=60": 5}
centrality_df["age_enc"] = centrality_df["agegrp9"].map(age_map).fillna(3)

# Normalize continuous features
scaler = MinMaxScaler()
cols_norm = [
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "weighted_degree_sec",
]
centrality_df[[c + "_norm" for c in cols_norm]] = scaler.fit_transform(
    centrality_df[cols_norm]
)

# Composite risk score
centrality_df["composite_risk_score"] = (
    0.35 * centrality_df["degree_centrality_norm"]
    + 0.30 * centrality_df["weighted_degree_sec_norm"]
    + 0.20 * centrality_df["betweenness_centrality_norm"]
    + 0.15 * centrality_df["sleep_room_enc"]
)

centrality_df = centrality_df.sort_values(
    ["composite_risk_score", "weighted_degree_sec", "node_id"],
    ascending=[False, False, True],
    kind="mergesort",
)
centrality_df["rank_by_composite"] = range(1, len(centrality_df) + 1)

centrality_df.to_csv("results/node_scores.csv", index=False)
print("OK: Exported results/node_scores.csv")
print("\nTop 15 highest-risk nodes:")
top_cols = [
    "node_id",
    "hhid",
    "sars",
    "is_index",
    "degree",
    "composite_risk_score",
    "rank_by_composite",
]
print(centrality_df.head(15)[top_cols].to_string(index=False))

# Validation: within top-k%, how many are SARS+?
for k in [0.01, 0.05, 0.10]:
    n_k = max(1, int(k * len(centrality_df)))
    top_k = centrality_df.head(n_k)
    pct_positive = (top_k["sars"] == "Positive").mean() * 100
    print(
        f"\nTop-{k*100:.0f}% ({n_k} nodes): {pct_positive:.1f}% are SARS+ "
        f"(baseline: {centrality_df['sars_label'].mean()*100:.1f}%)"
    )

logging.info("Risk score synthesis done")
