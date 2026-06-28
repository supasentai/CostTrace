"""Validate authoritative paper/result artifacts.

This is a lightweight reproducibility guard. It checks that declared source-of-truth
artifacts exist and expose the expected columns/fields, then reports retired values
that still appear in narrative surfaces without failing the command.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CsvArtifact:
    group: str
    path: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class JsonArtifact:
    group: str
    path: str
    fields: tuple[str, ...]


CSV_ARTIFACTS = (
    CsvArtifact(
        "model performance summary",
        "results/model/full_pytorch_graphsage_summary.csv",
        (
            "model",
            "runs",
            "test_auc_mean_std",
            "test_ap_mean_std",
            "test_f1_mean_std",
            "test_precision_mean_std",
            "test_recall_mean_std",
            "test_brier_mean_std",
        ),
    ),
    CsvArtifact(
        "benchmark table",
        "results/metrics/benchmark_table.csv",
        (
            "model",
            "feature_set",
            "runs",
            "test_auc_mean_std",
            "test_ap_mean_std",
            "test_f1_mean_std",
            "test_precision_mean_std",
            "test_recall_mean_std",
            "test_brier_mean_std",
        ),
    ),
    CsvArtifact(
        "ablation table",
        "results/metrics/ablation_table.csv",
        (
            "model",
            "feature_set",
            "runs",
            "test_auc_mean_std",
            "test_ap_mean_std",
            "test_f1_mean_std",
            "test_precision_mean_std",
            "test_recall_mean_std",
            "test_brier_mean_std",
        ),
    ),
    CsvArtifact(
        "intervention/strategy comparison",
        "results/intervention/final_comparison.csv",
        (
            "budget_k_pct",
            "budget_k_nodes",
            "strategy",
            "transmissions_blocked",
            "transmission_block_rate_pct",
            "prevention_rate_pct",
            "precision_k_pct",
            "transmission_coverage",
            "reduction_vs_baseline_pct",
        ),
    ),
    CsvArtifact(
        "SIR intervention details",
        "results/intervention/sir_intervention_results.csv",
        (
            "budget_k_pct",
            "strategy",
            "mean_infected_per_hh",
            "baseline_mean_infected",
            "reduction_vs_baseline_pct",
        ),
    ),
    CsvArtifact(
        "scenario comparison outputs",
        "results/metrics/trace_summary_table.csv",
        (
            "dataset",
            "method",
            "tasks",
            "hit_at_1",
            "hit_at_5",
            "mean_rank",
            "median_rank",
            "mrr",
            "mean_error_hops",
        ),
    ),
    CsvArtifact(
        "scenario dataset summary",
        "results/metrics/trace_dataset_summary.csv",
        ("dataset", "nodes", "edges", "source", "trace_tasks"),
    ),
)


JSON_ARTIFACTS = (
    JsonArtifact(
        "dataset summary",
        "data/processed/sashts/eda_summary.json",
        (
            "n_nodes",
            "n_edges",
            "n_raw_proximity_events",
            "n_components",
            "density",
            "avg_degree",
        ),
    ),
    JsonArtifact(
        "graph/network summary",
        "results/metrics/basic_metrics.json",
        (
            "n_nodes_total",
            "n_edges_total",
            "n_households",
            "avg_density_per_hh",
            "overall_attack_rate_pct",
        ),
    ),
    JsonArtifact(
        "final strategy summary",
        "results/intervention/final_strategy_summary.json",
        ("best_by_budget", "gnn_minus_random_prevention_rate_k1_pct", "n_rows"),
    ),
    JsonArtifact(
        "SIR baseline",
        "results/intervention/sir_baseline.json",
        (
            "baseline_mean_infected_per_hh",
            "observed_attack_rate_pct",
            "n_runs",
        ),
    ),
)


RETIRED_TOKENS = (
    "0.7669",
    "0.8995",
    "26.8",
    "49.0",
    "22.2",
    "42.7",
    "0.5842",
    "0.7809",
    "sir_results.csv",
)

SCAN_ROOTS = ("README.md", "docs", "reports", "notebooks")
SKIP_STALE_SCAN = {
    Path("docs/paper_upgrade/authoritative_artifacts.md"),
    Path("docs/paper_upgrade/metric_consistency_issues.md"),
}


def rel(path: Path) -> Path:
    return path.relative_to(ROOT)


def validate_csv(artifact: CsvArtifact) -> list[str]:
    path = ROOT / artifact.path
    errors: list[str] = []
    if not path.exists():
        return [f"missing {artifact.group}: {artifact.path}"]

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [column for column in artifact.columns if column not in columns]
        if missing:
            errors.append(
                f"{artifact.path} missing expected columns: {', '.join(missing)}"
            )
    return errors


def validate_json(artifact: JsonArtifact) -> list[str]:
    path = ROOT / artifact.path
    errors: list[str] = []
    if not path.exists():
        return [f"missing {artifact.group}: {artifact.path}"]

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    missing = [field for field in artifact.fields if field not in data]
    if missing:
        errors.append(f"{artifact.path} missing expected fields: {', '.join(missing)}")
    return errors


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    for scan_root in SCAN_ROOTS:
        path = ROOT / scan_root
        if path.is_file():
            files.append(path)
            continue
        if not path.exists():
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {
                ".md",
                ".tex",
                ".ipynb",
                ".txt",
            }:
                files.append(candidate)
    return files


def is_contextual_non_issue(line: str, hits: list[str]) -> bool:
    if hits == ["22.2"] and "Degree" in line:
        return True
    if hits == ["22.2"] and "Thêm smoke test" in line:
        return True
    return False


def report_stale_references() -> int:
    count = 0
    for path in iter_text_files():
        relative = rel(path)
        if relative in SKIP_STALE_SCAN:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            hits = [token for token in RETIRED_TOKENS if token in line]
            if hits and is_contextual_non_issue(line, hits):
                continue
            if hits:
                count += 1
                print(
                    "STALE_REFERENCE_REMAINING "
                    f"{relative}:{line_number}: {', '.join(hits)}"
                )
    return count


def main() -> int:
    errors: list[str] = []
    for artifact in CSV_ARTIFACTS:
        errors.extend(validate_csv(artifact))
    for artifact in JSON_ARTIFACTS:
        errors.extend(validate_json(artifact))

    stale_count = report_stale_references()
    if stale_count:
        print(f"STALE_REFERENCE_REMAINING_COUNT {stale_count}")

    if errors:
        for error in errors:
            print(f"AUTHORITATIVE_ARTIFACT_ERROR {error}")
        return 1

    print("AUTHORITATIVE_ARTIFACTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
