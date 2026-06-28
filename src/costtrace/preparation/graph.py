import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from costtrace.config import PATHS, require_existing


PATHS.gephi.mkdir(parents=True, exist_ok=True)
PATHS.processed_sashts.mkdir(parents=True, exist_ok=True)
edges = pd.read_csv(require_existing(PATHS.processed_edges_clean, "processed SASHTS edges"))
meta = pd.read_csv(require_existing(PATHS.processed_metadata, "processed SASHTS metadata"))

# ── 1. Build weighted graph ────────────────────────────────────────────────────
G = nx.from_pandas_edgelist(
    edges,
    source="indid1",
    target="indid2",
    edge_attr=["total_duration_sec", "n_contacts", "pair_sars", "has_real_ts"],
)

# ── 2. Gắn node attributes từ metadata ───────────────────────────────────────
meta_idx = meta.set_index("indid")
node_attrs = [
    "site",
    "agegrp9",
    "sex",
    "sars",
    "index",
    "sus",
    "hhid",
    "bmicat",
    "smokecignow1",
    "ixesarsvarf1",
    "sleep_room_ix",
    "cared_by_ix",
    "hh_ar",
]
for node in G.nodes():
    if node in meta_idx.index:
        for attr in node_attrs:
            try:
                G.nodes[node][attr] = meta_idx.loc[node, attr]
            except KeyError:
                pass

# ── 3. Export NodeList ─────────────────────────────────────────────────────────
node_records = []
for node in G.nodes(data=True):
    rec = {"node_id": node[0]}
    rec.update(node[1])
    rec["degree"] = G.degree(node[0])
    rec["weighted_degree"] = sum(
        d["total_duration_sec"] for _, _, d in G.edges(node[0], data=True)
    )
    node_records.append(rec)
nodelist_df = pd.DataFrame(node_records)
nodelist_df.to_csv(PATHS.processed_nodelist_canonical, index=False)

# ── 4. Export EdgeList ─────────────────────────────────────────────────────────
edgelist_df = edges[
    ["indid1", "indid2", "total_duration_sec", "n_contacts", "pair_sars"]
].copy()
edgelist_df.columns = ["source", "target", "weight", "n_contacts", "transmission"]
edgelist_df.to_csv(PATHS.processed_edgelist_canonical, index=False)

# ── 5. Lưu graph object ───────────────────────────────────────────────────────
with open(PATHS.processed_graph_canonical, "wb") as f:
    pickle.dump(G, f)
nx.write_gexf(G, PATHS.gephi / "contact_network.gexf")

# ── 6. Print stats ────────────────────────────────────────────────────────────
components = list(nx.connected_components(G))
print(f"Nodes              : {G.number_of_nodes()}")
print(f"Edges              : {G.number_of_edges()}")
print(f"Components         : {len(components)}  (= 88 households)")
print(f"Largest component  : {max(len(c) for c in components)} nodes")
print(f"Density            : {nx.density(G):.5f}")
degrees = [d for _, d in G.degree()]
print(f"Avg degree         : {np.mean(degrees):.2f}")
print(f"Max degree         : {max(degrees)}")
