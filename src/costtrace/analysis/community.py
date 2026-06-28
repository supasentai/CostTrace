import json
import logging
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import pickle
from networkx.algorithms.community import louvain_communities

from costtrace.config import PATHS, require_existing

log_path = Path("logs/analysis.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
)

logging.info("Community detection start")

with open(require_existing(PATHS.processed_graph, "processed SASHTS graph"), "rb") as f:
    G = pickle.load(f)

# Global Louvain (whole graph, weight = duration)
communities = sorted(
    [set(community) for community in louvain_communities(G, weight="total_duration_sec", seed=42)],
    key=lambda community: min(community),
)
modularity = nx.community.modularity(G, communities, weight="total_duration_sec")

print("=== LOUVAIN COMMUNITY DETECTION ===")
print(f"  So community tim duoc : {len(communities)}")
print(f"  Modularity score      : {modularity:.4f}")
print("  (Ky vong: ~88 vi 88 HH components, validate cau truc)")

# Map node to community id
node_community = {}
for cid, comm in enumerate(communities):
    for node in sorted(comm):
        node_community[node] = cid

# Compare with household label
centrality_df = pd.read_csv(PATHS.metrics / "centrality_scores.csv")
centrality_df["louvain_community_id"] = centrality_df["node_id"].map(node_community)
centrality_df = centrality_df.sort_values(["hhid", "node_id"]).reset_index(drop=True)

# Agreement: within-HH pairs in same community
agreement_count = 0
total_pairs = 0
for hhid, group in centrality_df.groupby("hhid"):
    nodes_in_hh = group["node_id"].tolist()
    for a, b in combinations(nodes_in_hh, 2):
        total_pairs += 1
        if node_community.get(a) == node_community.get(b):
            agreement_count += 1

agreement_pct = agreement_count / total_pairs * 100 if total_pairs > 0 else 0
print(
    f"\n  HH-Community agreement: {agreement_pct:.1f}% of within-HH pairs share same community"
)
print("  (100% = Louvain perfectly recovers HH structure)")

# Export
PATHS.metrics.mkdir(parents=True, exist_ok=True)
centrality_df.to_csv(PATHS.metrics / "centrality_scores.csv", index=False)

community_df = pd.DataFrame(list(node_community.items()), columns=["node_id", "community_id"])
community_df["community_size"] = community_df["community_id"].map(
    community_df["community_id"].value_counts()
)
community_df = community_df.sort_values(["community_id", "node_id"]).reset_index(drop=True)
community_df.to_csv(PATHS.metrics / "community_assignments.csv", index=False)

with open(PATHS.metrics / "community_metrics.json", "w") as f:
    json.dump(
        {
            "method": "louvain",
            "weight": "total_duration_sec",
            "n_communities": len(communities),
            "modularity": modularity,
            "hh_agreement_pct": round(agreement_pct, 1),
            "community_sizes": sorted([len(c) for c in communities], reverse=True),
        },
        f,
        indent=2,
    )

print(
    f"\nCommunity analysis completed. Communities: {len(communities)}, "
    f"Modularity: {modularity:.4f}, HH-agreement: {agreement_pct:.1f}%"
)

logging.info(
    "Community detection done | communities=%s modularity=%.4f agreement=%.1f",
    len(communities),
    modularity,
    agreement_pct,
)
