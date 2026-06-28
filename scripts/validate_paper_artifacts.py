from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from costtrace.config import PATHS


PAPER_ROOT = PATHS.results / "paper_ready"
FIGURE_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"
DOC_DIR = PATHS.root / "docs" / "paper_upgrade"

REQUIRED_FIGURE_IDS = {
    "fig01_network_overview",
    "fig02_degree_distribution",
    "fig03_model_comparison",
    "fig04_ablation_results",
    "fig05_budget_curves",
    "fig06_uncertainty_plots",
    "fig07_parameter_sweep",
    "fig08_scenario_reduction",
    "fig09_time_series",
}

REQUIRED_TABLE_IDS = {
    "table01_dataset_summary",
    "table02_graph_summary",
    "table03_model_benchmark",
    "table04_ablation",
    "table05_strategy_comparison",
    "table06_budget_analysis",
    "table07_uncertainty_summary",
    "table08_parameter_sensitivity",
    "table09_supplementary_full_metrics",
}

STALE_FILENAMES = ("sir_results.csv",)


def split_sources(value: str) -> list[Path]:
    return [PATHS.root / part.strip() for part in str(value).split(";") if part.strip()]


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_figures() -> pd.DataFrame:
    manifest_path = FIGURE_DIR / "figure_manifest.csv"
    require_file(manifest_path, "figure manifest")
    manifest = pd.read_csv(manifest_path)
    missing_ids = REQUIRED_FIGURE_IDS - set(manifest["figure_id"])
    if missing_ids:
        raise ValueError(f"Missing required figures in manifest: {sorted(missing_ids)}")
    for _, row in manifest.iterrows():
        require_file(PATHS.root / row["filename"], f"figure {row['figure_id']}")
        for source in split_sources(row["source_artifact"]):
            require_file(source, f"source artifact for {row['figure_id']}")
        if row["source_script"] != "scripts/31_generate_paper_artifacts.py":
            raise ValueError(f"Unexpected source script for {row['figure_id']}")
    return manifest


def validate_tables() -> pd.DataFrame:
    manifest_path = TABLE_DIR / "table_manifest.csv"
    require_file(manifest_path, "table manifest")
    manifest = pd.read_csv(manifest_path)
    missing_ids = REQUIRED_TABLE_IDS - set(manifest["table_id"])
    if missing_ids:
        raise ValueError(f"Missing required tables in manifest: {sorted(missing_ids)}")
    for _, row in manifest.iterrows():
        for column in ["csv_path", "markdown_path", "latex_path"]:
            require_file(PATHS.root / row[column], f"{column} for {row['table_id']}")
        for source in split_sources(row["source_artifact"]):
            require_file(source, f"source artifact for {row['table_id']}")
        if row["source_script"] != "scripts/31_generate_paper_artifacts.py":
            raise ValueError(f"Unexpected source script for {row['table_id']}")
    return manifest


def validate_table_consistency() -> None:
    benchmark_source = pd.read_csv(PATHS.metrics / "benchmark_table.csv")
    benchmark_table = pd.read_csv(TABLE_DIR / "table03_model_benchmark.csv")
    if len(benchmark_source) != len(benchmark_table):
        raise ValueError("table03_model_benchmark row count does not match benchmark_table.csv")
    if benchmark_source["test_auc_mean_std"].tolist() != benchmark_table["test_auc_mean_std"].tolist():
        raise ValueError("table03_model_benchmark values differ from benchmark_table.csv")

    ablation_source = pd.read_csv(PATHS.metrics / "ablation_table.csv")
    ablation_table = pd.read_csv(TABLE_DIR / "table04_ablation.csv")
    if len(ablation_source) != len(ablation_table):
        raise ValueError("table04_ablation row count does not match ablation_table.csv")

    final_source = pd.read_csv(PATHS.intervention / "final_comparison.csv").sort_values(
        ["budget_k_pct", "strategy"]
    )
    strategy_table = pd.read_csv(TABLE_DIR / "table05_strategy_comparison.csv").sort_values(
        ["budget_k_pct", "strategy"]
    )
    for column in ["prevention_rate_pct", "reduction_vs_baseline_pct", "transmission_coverage"]:
        if final_source[column].round(4).tolist() != strategy_table[column].round(4).tolist():
            raise ValueError(f"table05_strategy_comparison column {column} differs from source")

    budget_source = pd.read_csv(PATHS.intervention / "phase05_budget_decision_table.csv")
    budget_table = pd.read_csv(TABLE_DIR / "table06_budget_analysis.csv")
    if len(budget_source) != len(budget_table):
        raise ValueError("table06_budget_analysis row count differs from phase05_budget_decision_table.csv")

    eda = json.loads(PATHS.eda_summary.read_text(encoding="utf-8"))
    dataset_table = pd.read_csv(TABLE_DIR / "table01_dataset_summary.csv")
    dataset_values = dataset_table.set_index("metric")["value"].to_dict()
    for metric in ["n_raw_proximity_events", "n_nodes", "n_edges", "n_components"]:
        if str(eda[metric]) != str(dataset_values[metric]):
            raise ValueError(f"table01_dataset_summary metric {metric} differs from EDA summary")


def validate_docs_and_stale_filenames() -> None:
    for path in [
        FIGURE_DIR / "figure_manifest.csv",
        TABLE_DIR / "table_manifest.csv",
        DOC_DIR / "artifact_traceability.md",
        DOC_DIR / "consistency_report.md",
    ]:
        require_file(path, "paper documentation")
        text = path.read_text(encoding="utf-8")
        for stale in STALE_FILENAMES:
            if stale in text:
                raise ValueError(f"Stale filename {stale} found in {path}")


def main() -> int:
    validate_figures()
    validate_tables()
    validate_table_consistency()
    validate_docs_and_stale_filenames()
    print("PAPER_ARTIFACTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
