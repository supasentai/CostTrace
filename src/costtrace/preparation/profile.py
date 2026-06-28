import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from costtrace.config import PATHS, require_existing


with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
    G = pickle.load(f)
meta = pd.read_csv(require_existing(PATHS.processed_metadata, "processed SASHTS metadata"))
edges = pd.read_csv(require_existing(PATHS.processed_edgelist, "processed SASHTS edgelist"))

# Per-household stats
comps = list(nx.connected_components(G))
hh_sizes = sorted([len(c) for c in comps])

# SARS stats
sars_pos = (meta["sars"] == "Positive").sum()
sars_neg = (meta["sars"] == "Negative").sum()

# Transmission edges
trans_edges = (edges["transmission"] == "Transmission").sum()

eda = {
    "dataset_name": "SASHTS – South Africa Household Transmission Study",
    "files": ["sashts_contact_network.csv", "sashts_metadata.csv"],
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "n_raw_proximity_events": 140542,
    "is_directed": False,
    "is_weighted": True,
    "weight_column": "total_duration_sec (aggregated contact duration per pair)",
    "has_timestamp": "Partial – Soweto only (no_ts=0)",
    "n_components": len(comps),
    "component_note": "Each component = 1 household (no cross-HH edges)",
    "household_size_dist": {str(s): hh_sizes.count(s) for s in set(hh_sizes)},
    "avg_household_size": float(np.mean(hh_sizes)),
    "density": nx.density(G),
    "avg_degree": float(np.mean([d for _, d in G.degree()])),
    "max_degree": int(max(d for _, d in G.degree())),
    "n_sars_positive": int(sars_pos),
    "n_sars_negative": int(sars_neg),
    "attack_rate_overall_pct": round(sars_pos / G.number_of_nodes() * 100, 1),
    "n_index_cases": int((meta["index"] == "Index").sum()),
    "n_transmission_edges": int(trans_edges),
    "n_non_transmission_edges": int(len(edges) - trans_edges),
    "sites": {"Soweto": 197, "Klerksdorp": 143},
    "variants": meta["ixesarsvarf1"].value_counts().dropna().to_dict(),
    "node_features_available": [
        "site",
        "agegrp9",
        "sex",
        "sars",
        "index",
        "sus",
        "bmicat",
        "smokecignow1",
        "sleep_room_ix",
        "cared_by_ix",
        "ixesarsvarf1",
        "hh_ar",
    ],
}

# Sửa đổi: ghi JSON bằng UTF-8 để giữ đúng ký tự Unicode trong dataset_name/has_timestamp.
with open(PATHS.eda_summary_canonical, "w", encoding="utf-8") as f:
    json.dump(eda, f, indent=2, default=str)

print(json.dumps(eda, indent=2, default=str))
print("\nData profile completed")
print(
    f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()} | Households: {len(comps)}"
)
