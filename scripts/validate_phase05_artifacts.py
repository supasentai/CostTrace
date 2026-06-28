from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from costtrace.config import PATHS, phase05_config


CFG = phase05_config("smoke")

EXPECTED_COLUMNS = {
    CFG.run_level_results: {
        "profile",
        "param_id",
        "beta",
        "gamma",
        "run_id",
        "seed",
        "budget_k_pct",
        "budget_k_nodes",
        "strategy",
        "selected_node_ids",
        "final_mean_infected_per_hh",
        "baseline_mean_infected_per_hh",
        "reduction_vs_baseline_pct",
        "prevention_rate_pct",
        "precision_k_pct",
        "transmission_coverage",
    },
    CFG.timeseries_results: {
        "profile",
        "param_id",
        "beta",
        "gamma",
        "run_id",
        "seed",
        "budget_k_pct",
        "budget_k_nodes",
        "strategy",
        "time_step",
        "mean_susceptible",
        "mean_infected_active",
        "mean_recovered",
        "mean_ever_infected",
        "n_households",
    },
    CFG.baseline_summary: {
        "profile",
        "param_id",
        "beta",
        "gamma",
        "n_runs",
        "mean_baseline_infected_per_hh",
        "std_baseline_infected_per_hh",
        "p2_5_baseline_infected_per_hh",
        "p97_5_baseline_infected_per_hh",
    },
    CFG.parameter_sweep: {
        "profile",
        "param_id",
        "beta",
        "gamma",
        "strategy",
        "budget_k_pct",
        "mean_infected_per_hh",
        "mean_reduction_vs_baseline_pct",
        "p2_5_reduction_vs_baseline_pct",
        "p97_5_reduction_vs_baseline_pct",
        "n_runs",
    },
    CFG.budget_curve: {
        "profile",
        "beta",
        "gamma",
        "strategy",
        "budget_k_pct",
        "budget_k_nodes",
        "mean_infected_per_hh",
        "reduction_vs_baseline_pct",
        "prevention_rate_pct",
        "transmission_coverage",
        "marginal_sir_reduction_gain_pct",
        "n_runs",
    },
    CFG.budget_decision_table: {
        "budget_k_pct",
        "budget_k_nodes",
        "best_strategy_by_prevention",
        "best_prevention_rate_pct",
        "best_strategy_by_sir_reduction",
        "best_sir_reduction_pct",
        "best_strategy_by_transmission_coverage",
        "best_transmission_coverage_pct",
    },
    CFG.uncertainty_summary: {
        "profile",
        "strategy",
        "budget_k_pct",
        "budget_k_nodes",
        "metric_name",
        "point_estimate",
        "lower_interval",
        "upper_interval",
        "method",
        "n_resamples_or_runs",
    },
}


UNIQUE_KEYS = {
    CFG.run_level_results: ["param_id", "run_id", "budget_k_pct", "strategy"],
    CFG.timeseries_results: ["param_id", "run_id", "budget_k_pct", "strategy", "time_step"],
    CFG.parameter_sweep: ["param_id", "strategy", "budget_k_pct"],
    CFG.budget_curve: ["strategy", "budget_k_pct"],
    CFG.budget_decision_table: ["budget_k_pct"],
    CFG.uncertainty_summary: ["strategy", "budget_k_pct", "metric_name"],
}


FIGURE_SOURCES = {
    "phase05_timeseries_curve.png": CFG.timeseries_results,
    "phase05_budget_curve.png": CFG.budget_curve,
    "phase05_parameter_sweep.png": CFG.parameter_sweep,
    "phase05_uncertainty_intervals.png": CFG.uncertainty_summary,
}


def read_artifact(filename: str) -> pd.DataFrame:
    path = PATHS.intervention / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 05 artifact: {path}")
    df = pd.read_csv(path)
    missing = sorted(EXPECTED_COLUMNS[filename] - set(df.columns))
    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{filename} is empty")
    return df


def validate_budget_range(filename: str, df: pd.DataFrame) -> None:
    if "budget_k_pct" not in df:
        return
    bad = df[(df["budget_k_pct"] < 0) | (df["budget_k_pct"] > 100)]
    if not bad.empty:
        raise ValueError(f"{filename} has budget values outside 0..100")


def validate_uniqueness(filename: str, df: pd.DataFrame) -> None:
    keys = UNIQUE_KEYS.get(filename)
    if not keys:
        return
    duplicated = df.duplicated(keys)
    if duplicated.any():
        raise ValueError(f"{filename} has duplicate key rows for {keys}: {int(duplicated.sum())}")


def validate_intervals(filename: str, df: pd.DataFrame) -> None:
    interval_sets = [
        ("p2_5_reduction_vs_baseline_pct", "mean_reduction_vs_baseline_pct", "p97_5_reduction_vs_baseline_pct"),
        ("p2_5_infected_per_hh", "mean_infected_per_hh", "p97_5_infected_per_hh"),
        ("p2_5_baseline_infected_per_hh", "mean_baseline_infected_per_hh", "p97_5_baseline_infected_per_hh"),
        ("lower_interval", "point_estimate", "upper_interval"),
    ]
    for lower, point, upper in interval_sets:
        if {lower, point, upper}.issubset(df.columns):
            invalid = df[(df[lower] > df[point]) | (df[point] > df[upper])]
            if not invalid.empty:
                raise ValueError(f"{filename} has invalid interval ordering for {point}")


def validate_figures() -> None:
    if not PATHS.phase05_figures.exists():
        return
    for figure, source in FIGURE_SOURCES.items():
        figure_path = PATHS.phase05_figures / figure
        if figure_path.exists() and not (PATHS.intervention / source).exists():
            raise ValueError(f"{figure_path} exists without source {source}")
    if (PATHS.phase05_figures / "phase05_timeseries_curve.png").exists():
        timeseries = pd.read_csv(PATHS.intervention / CFG.timeseries_results)
        if timeseries.empty:
            raise ValueError("time-series figure exists but time-series CSV is empty")


def main() -> int:
    for filename in EXPECTED_COLUMNS:
        df = read_artifact(filename)
        validate_budget_range(filename, df)
        validate_uniqueness(filename, df)
        validate_intervals(filename, df)
    validate_figures()
    print("PHASE05_ARTIFACTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
