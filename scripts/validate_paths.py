from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from costtrace.config import PATHS  # noqa: E402


EXPECTED_ROWS = {
    "raw contact table": (PATHS.raw_contact_network_canonical, 140_542),
    "raw metadata table": (PATHS.raw_metadata_canonical, 340),
    "processed node table": (PATHS.processed_nodelist_canonical, 340),
    "processed edge table": (PATHS.processed_edgelist_canonical, 542),
}

REQUIRED_ARTIFACTS = {
    "processed clean edge table": PATHS.processed_edges_clean_canonical,
    "processed graph": PATHS.processed_graph_canonical,
}


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected file for {label}: {path}")


def main() -> None:
    print("=== COSTTRACE PATH VALIDATION ===")
    print(f"Repository root: {PATHS.root}")

    for label, path in REQUIRED_ARTIFACTS.items():
        require_file(path, label)
        print(f"OK {label}: {path}")

    for label, (path, expected) in EXPECTED_ROWS.items():
        require_file(path, label)
        observed = count_csv_rows(path)
        if observed != expected:
            raise AssertionError(
                f"{label} row count mismatch: expected {expected}, got {observed} ({path})"
            )
        print(f"OK {label}: {observed:,} rows at {path}")

    print("PATH_VALIDATION_OK")


if __name__ == "__main__":
    main()
