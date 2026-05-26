import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pickle

log_path = Path("logs/phase02.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
)

logging.info("Task 6: centrality scores start")

G = pickle.load(open("data/processed/graph.pkl", "rb"))
comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]

all_centrality = []

for sg in comps:
    n = sg.number_of_nodes()
    if n < 2:
        continue

    # Degree centrality (normalized per household)
    dc = nx.degree_centrality(sg)
    # Degree raw (contact count in household)
    deg = dict(sg.degree())
    # Weighted degree = total contact duration
    wdeg = {
        node: sum(d["total_duration_sec"] for _, _, d in sg.edges(node, data=True))
        for node in sg.nodes()
    }
    # Betweenness (exact, household size is small)
    bc = nx.betweenness_centrality(sg, normalized=True, weight="total_duration_sec")
    # Closeness
    cc = nx.closeness_centrality(sg, distance="total_duration_sec")

    for node in sg.nodes():
        attrs = sg.nodes[node]
        all_centrality.append(
            {
                "node_id": node,
                "hhid": attrs.get("hhid", ""),
                "site": attrs.get("site", ""),
                "sars": attrs.get("sars", ""),
                "is_index": 1 if attrs.get("index") == "Index" else 0,
                "agegrp9": attrs.get("agegrp9", ""),
                "sex": attrs.get("sex", ""),
                "degree": deg[node],
                "degree_centrality": dc[node],
                "weighted_degree_sec": wdeg[node],
                "betweenness_centrality": bc[node],
                "closeness_centrality": cc[node],
                "sleep_room_ix": attrs.get("sleep_room_ix", ""),
                "cared_by_ix": attrs.get("cared_by_ix", ""),
                "sus": attrs.get("sus", ""),
                "bmicat": attrs.get("bmicat", ""),
                "hh_ar": attrs.get("hh_ar", 0),
            }
        )

centrality_df = pd.DataFrame(all_centrality)
centrality_df.to_csv("results/centrality_scores.csv", index=False)

# Print top super-spreaders (SARS+ nodes with highest degree)
print("\n=== TOP 10 SUPER-SPREADERS (SARS+ | Highest Degree Centrality) ===")
top = (
    centrality_df[centrality_df["sars"] == "Positive"]
    .sort_values("degree_centrality", ascending=False)
    .head(10)
)
print(
    top[
        [
            "node_id",
            "hhid",
            "degree",
            "degree_centrality",
            "weighted_degree_sec",
            "hh_ar",
        ]
    ].to_string(index=False)
)

print("\n=== TOP 10 BRIDGE NODES (Highest Betweenness) ===")
top_bc = centrality_df.sort_values("betweenness_centrality", ascending=False).head(10)
print(
    top_bc[["node_id", "hhid", "sars", "degree", "betweenness_centrality"]].to_string(
        index=False
    )
)

logging.info("Task 6: centrality scores done")
