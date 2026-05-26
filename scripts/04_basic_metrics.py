import json
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

logging.info("Task 5: basic network metrics start")

G = pickle.load(open("data/processed/graph.pkl", "rb"))
comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]

# Per-component stats
comp_stats = []
for i, sg in enumerate(comps):
    comp_stats.append(
        {
            "hh": list(sg.nodes(data=True))[0][1].get("hhid", ""),
            "n_nodes": sg.number_of_nodes(),
            "n_edges": sg.number_of_edges(),
            "density": nx.density(sg),
            "diameter": nx.diameter(sg) if sg.number_of_nodes() > 1 else 0,
            "avg_clustering": nx.average_clustering(sg),
            "sars_positive": sum(
                1 for _, d in sg.nodes(data=True) if d.get("sars") == "Positive"
            ),
            "hh_ar": list(sg.nodes(data=True))[0][1].get("hh_ar", 0),
        }
    )

comp_df = pd.DataFrame(comp_stats)
comp_df.to_csv("results/household_metrics.csv", index=False)

# Aggregate metrics
metrics = {
    "n_nodes_total": G.number_of_nodes(),
    "n_edges_total": G.number_of_edges(),
    "n_households": len(comps),
    "avg_household_size": float(comp_df["n_nodes"].mean()),
    "avg_density_per_hh": float(comp_df["density"].mean()),
    "avg_diameter_per_hh": float(comp_df["diameter"].mean()),
    "avg_clustering_per_hh": float(comp_df["avg_clustering"].mean()),
    "overall_attack_rate_pct": float(
        sum(comp_df["sars_positive"]) / G.number_of_nodes() * 100
    ),
    "note": "Metrics computed per household component (88 separate HH subgraphs)",
}

with open("results/basic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("=== BASIC NETWORK METRICS ===")
for k, v in metrics.items():
    print(f"  {k:<35}: {v}")

logging.info("Task 5: basic network metrics done")
