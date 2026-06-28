import os
import sys

import numpy as np
import pandas as pd

from costtrace.config import PATHS, require_existing


sys.stdout.reconfigure(encoding="utf-8")

net = pd.read_csv(require_existing(PATHS.raw_contact_network, "raw SASHTS contact network"))
meta = pd.read_csv(require_existing(PATHS.raw_metadata, "raw SASHTS metadata"))

print(f"Raw rows: {len(net):,}")

#  1. Kiểm tra null 
print("\nNull counts (contact_network):")
print(net.isnull().sum()[net.isnull().sum() > 0])
# Kết quả mong đợi: 0 null trong tất cả cột

#  2. Loại bỏ self-loop (nếu có) 
self_loops = net[net["indid1"] == net["indid2"]]
print(f"\nSelf-loops: {len(self_loops)}")
net = net[net["indid1"] != net["indid2"]].copy()

#  3. Đảm bảo pair không bị đảo thứ tự (A-B và B-A là cùng 1 cặp) 
net["pair_sorted"] = net.apply(
    lambda r: "_".join(sorted([r["indid1"], r["indid2"]])), axis=1
)

swap_mask = net["indid1"] > net["indid2"]
for col1, col2 in [
    ("indid1", "indid2"),
    ("sars_indid1", "sars_indid2"),
    ("age_indid1", "age_indid2"),
]:
    left_values = net.loc[swap_mask, col1].copy()
    net.loc[swap_mask, col1] = net.loc[swap_mask, col2].values
    net.loc[swap_mask, col2] = left_values.values

#  4. Flag Klerksdorp (no real timestamp) 
# no_ts=1 → timestamp là 1970-01-xx (fake)
# no_ts=0 → timestamp thực từ Soweto
net["has_real_ts"] = net["no_ts"] == 0
print(f"\nRows with real timestamp : {net['has_real_ts'].sum():,}")
print(f"Rows without timestamp   : {(~net['has_real_ts']).sum():,}")

#  5. Aggregate thành weighted edges 
edge_agg = (
    net.groupby(["indid1", "indid2", "hh"])
    .agg(
        pair_sars=("pair_sars", "first"),
        sars_indid1=("sars_indid1", "first"),
        sars_indid2=("sars_indid2", "first"),
        age_indid1=("age_indid1", "first"),
        age_indid2=("age_indid2", "first"),
        total_duration_sec=("duration_sec", "sum"),
        n_contacts=("duration_sec", "count"),
        has_real_ts=("has_real_ts", "any"),
    )
    .reset_index()
)
print(f"\nAggregated edges: {len(edge_agg)}")
print(f"Expected unique pairs: 542 | Got: {len(edge_agg)}")

#  6. Validate: tất cả node trong edge list phải có trong metadata 
edge_nodes = set(edge_agg["indid1"]) | set(edge_agg["indid2"])
meta_nodes = set(meta["indid"])
diff = edge_nodes - meta_nodes
print(f"\nNodes in edges but NOT in metadata: {len(diff)}")  # phải = 0

#  7. Export 
os.makedirs(PATHS.processed_sashts, exist_ok=True)

edge_agg.to_csv(PATHS.processed_edges_clean_canonical, index=False)
meta.to_csv(PATHS.processed_metadata_canonical, index=False)

print("\nCleaning done")
print(f"   Exported: {PATHS.processed_edges_clean_canonical} ({len(edge_agg)} edges)")
print(f"   Exported: {PATHS.processed_metadata_canonical} ({len(meta)} nodes)")
