import pandas as pd
import sys

from costtrace.config import PATHS, require_existing


sys.stdout.reconfigure(encoding="utf-8")

net = pd.read_csv(require_existing(PATHS.raw_contact_network, "raw SASHTS contact network"))
meta = pd.read_csv(require_existing(PATHS.raw_metadata, "raw SASHTS metadata"))

assert len(net) == 140542, "contact_network row count mismatch"
assert len(meta) == 340, "metadata row count mismatch"
assert net["indid1"].nunique() == 340
assert meta["indid"].nunique() == 340

print("✅ Data verified")
print(f"   contact_network : {len(net):,} rows | {net['pair'].nunique()} unique pairs")
print(f"   metadata        : {len(meta)} individuals | {meta['hhid'].nunique()} households")
