import pandas as pd
import sys


sys.stdout.reconfigure(encoding="utf-8")

net = pd.read_csv("data/raw/sashts_contact_network.csv")
meta = pd.read_csv("data/raw/sashts_metadata.csv")

assert len(net) == 140542, "contact_network row count mismatch"
assert len(meta) == 340, "metadata row count mismatch"
assert net["indid1"].nunique() == 340
assert meta["indid"].nunique() == 340

print("✅ Data verified")
print(f"   contact_network : {len(net):,} rows | {net['pair'].nunique()} unique pairs")
print(f"   metadata        : {len(meta)} individuals | {meta['hhid'].nunique()} households")
