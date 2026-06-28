from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from costtrace.config import PATHS, require_existing


SCRIPT = "scripts/31_generate_paper_artifacts.py"
PAPER_ROOT = PATHS.results / "paper_ready"
FIGURE_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"
DOC_DIR = PATHS.root / "docs" / "paper_upgrade"

STYLE_COLORS = {
    "random": "#6B7280",
    "degree": "#2563EB",
    "betweenness": "#D97706",
    "gnn": "#059669",
    "baseline": "#111827",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(PATHS.root).as_posix()


def setup_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    text = dataframe_to_markdown(df)
    path.write_text(text + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(format_cell)
    columns = list(display.columns)
    rows = display.values.tolist()
    widths = [
        max(len(str(col)), *(len(str(row[idx])) for row in rows)) if rows else len(str(col))
        for idx, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return ""
    return str(value)


def write_table(df: pd.DataFrame, table_id: str) -> dict[str, str]:
    csv_path = TABLE_DIR / f"{table_id}.csv"
    md_path = TABLE_DIR / f"{table_id}.md"
    tex_path = TABLE_DIR / f"{table_id}.tex"
    df.to_csv(csv_path, index=False)
    save_markdown_table(df, md_path)
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(df.to_latex(index=False, escape=True))
    return {
        "csv": rel(csv_path),
        "markdown": rel(md_path),
        "latex": rel(tex_path),
    }


def save_figure(fig: plt.Figure, filename: str) -> str:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rel(path)


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
        }
    )


def build_tables(timestamp: str) -> list[dict[str, str]]:
    table_specs: list[dict[str, str]] = []

    eda = json.loads(require_existing(PATHS.eda_summary, "EDA summary").read_text(encoding="utf-8"))
    dataset_fields = [
        "dataset_name",
        "n_raw_proximity_events",
        "n_nodes",
        "n_edges",
        "n_components",
        "avg_household_size",
        "density",
        "avg_degree",
        "n_sars_positive",
        "n_sars_negative",
        "attack_rate_overall_pct",
    ]
    dataset_df = pd.DataFrame(
        [{"metric": field, "value": eda.get(field)} for field in dataset_fields]
    )
    table_specs.append(
        table_row(
            "table01_dataset_summary",
            "Dataset summary",
            write_table(dataset_df, "table01_dataset_summary"),
            rel(PATHS.eda_summary),
            "Methods: Dataset",
            timestamp,
        )
    )

    basic = json.loads((PATHS.metrics / "basic_metrics.json").read_text(encoding="utf-8"))
    graph_fields = [
        "n_nodes_total",
        "n_edges_total",
        "n_households",
        "avg_household_size",
        "avg_density_per_hh",
        "avg_diameter_per_hh",
        "avg_clustering_per_hh",
        "overall_attack_rate_pct",
    ]
    graph_df = pd.DataFrame([{"metric": field, "value": basic.get(field)} for field in graph_fields])
    table_specs.append(
        table_row(
            "table02_graph_summary",
            "Graph summary",
            write_table(graph_df, "table02_graph_summary"),
            rel(PATHS.metrics / "basic_metrics.json"),
            "Results: Network structure",
            timestamp,
        )
    )

    benchmark = pd.read_csv(PATHS.metrics / "benchmark_table.csv")
    benchmark_table = benchmark[
        ["model", "feature_set", "runs", "test_auc_mean_std", "test_ap_mean_std", "test_f1_mean_std", "test_brier_mean_std"]
    ]
    table_specs.append(
        table_row(
            "table03_model_benchmark",
            "Model benchmark",
            write_table(benchmark_table, "table03_model_benchmark"),
            rel(PATHS.metrics / "benchmark_table.csv"),
            "Results: Model benchmark",
            timestamp,
        )
    )

    ablation = pd.read_csv(PATHS.metrics / "ablation_table.csv")
    ablation_table = ablation[
        ["model", "feature_set", "runs", "test_auc_mean_std", "test_ap_mean_std", "test_f1_mean_std", "test_brier_mean_std"]
    ]
    table_specs.append(
        table_row(
            "table04_ablation",
            "Ablation results",
            write_table(ablation_table, "table04_ablation"),
            rel(PATHS.metrics / "ablation_table.csv"),
            "Results: Ablation",
            timestamp,
        )
    )

    strategy = pd.read_csv(PATHS.intervention / "final_comparison.csv")
    strategy_table = strategy[
        [
            "budget_k_pct",
            "budget_k_nodes",
            "strategy",
            "prevention_rate_pct",
            "reduction_vs_baseline_pct",
            "precision_k_pct",
            "transmission_coverage",
        ]
    ].sort_values(["budget_k_pct", "strategy"])
    table_specs.append(
        table_row(
            "table05_strategy_comparison",
            "Strategy comparison",
            write_table(strategy_table, "table05_strategy_comparison"),
            rel(PATHS.intervention / "final_comparison.csv"),
            "Results: Intervention comparison",
            timestamp,
        )
    )

    budget = pd.read_csv(PATHS.intervention / "phase05_budget_decision_table.csv")
    table_specs.append(
        table_row(
            "table06_budget_analysis",
            "Budget analysis",
            write_table(budget, "table06_budget_analysis"),
            rel(PATHS.intervention / "phase05_budget_decision_table.csv"),
            "Results: Budget curves",
            timestamp,
        )
    )

    uncertainty = pd.read_csv(PATHS.intervention / "phase05_uncertainty_summary.csv")
    uncertainty_table = uncertainty[
        uncertainty["metric_name"].isin(["reduction_vs_baseline_pct", "prevention_rate_pct"])
    ].copy()
    table_specs.append(
        table_row(
            "table07_uncertainty_summary",
            "Uncertainty summary",
            write_table(uncertainty_table, "table07_uncertainty_summary"),
            rel(PATHS.intervention / "phase05_uncertainty_summary.csv"),
            "Results: Uncertainty",
            timestamp,
        )
    )

    sensitivity = pd.read_csv(PATHS.intervention / "phase05_parameter_sweep.csv")
    sensitivity_table = sensitivity[
        [
            "param_id",
            "beta",
            "gamma",
            "strategy",
            "budget_k_pct",
            "mean_reduction_vs_baseline_pct",
            "p2_5_reduction_vs_baseline_pct",
            "p97_5_reduction_vs_baseline_pct",
            "n_runs",
        ]
    ]
    table_specs.append(
        table_row(
            "table08_parameter_sensitivity",
            "Parameter sensitivity",
            write_table(sensitivity_table, "table08_parameter_sensitivity"),
            rel(PATHS.intervention / "phase05_parameter_sweep.csv"),
            "Supplement: Parameter sensitivity",
            timestamp,
        )
    )

    full_sources = [
        PATHS.model_results / "full_pytorch_graphsage_full_metrics.csv",
        PATHS.metrics / "benchmark_full_metrics.csv",
        PATHS.metrics / "ablation_full_metrics.csv",
    ]
    full_tables = []
    for source in full_sources:
        frame = pd.read_csv(source)
        frame.insert(0, "source_artifact", rel(source))
        full_tables.append(frame)
    full_metrics = pd.concat(full_tables, ignore_index=True, sort=False)
    table_specs.append(
        table_row(
            "table09_supplementary_full_metrics",
            "Supplementary full metrics",
            write_table(full_metrics, "table09_supplementary_full_metrics"),
            "; ".join(rel(path) for path in full_sources),
            "Supplement: Full metrics",
            timestamp,
        )
    )

    manifest = pd.DataFrame(table_specs)
    manifest.to_csv(TABLE_DIR / "table_manifest.csv", index=False)
    return table_specs


def table_row(
    table_id: str,
    title: str,
    paths: dict[str, str],
    source_artifact: str,
    manuscript_section: str,
    timestamp: str,
) -> dict[str, str]:
    return {
        "table_id": table_id,
        "title": title,
        "csv_path": paths["csv"],
        "markdown_path": paths["markdown"],
        "latex_path": paths["latex"],
        "source_artifact": source_artifact,
        "source_script": SCRIPT,
        "generation_timestamp_utc": timestamp,
        "manuscript_section": manuscript_section,
    }


def build_figures(timestamp: str) -> list[dict[str, str]]:
    figures: list[dict[str, str]] = []

    nodes = pd.read_csv(PATHS.processed_nodelist)
    edges = pd.read_csv(PATHS.processed_edgelist)
    graph = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=True)
    degree_lookup = nodes.set_index("node_id")["degree"].astype(float).to_dict()
    layout = nx.spring_layout(graph, seed=42, weight="weight", iterations=120)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    edge_widths = np.log1p(edges["weight"].astype(float))
    edge_widths = 0.25 + 1.75 * (edge_widths - edge_widths.min()) / max(edge_widths.max() - edge_widths.min(), 1)
    nx.draw_networkx_edges(graph, layout, ax=ax, width=edge_widths, alpha=0.25, edge_color="#6B7280")
    sizes = [24 + 18 * degree_lookup.get(node, 0.0) for node in graph.nodes()]
    colors = ["#DC2626" if nodes.set_index("node_id").loc[node, "sars"] == "Positive" else "#2563EB" for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, layout, ax=ax, node_size=sizes, node_color=colors, alpha=0.9, linewidths=0.2, edgecolors="white")
    ax.set_title("SASHTS Household Contact Network")
    ax.set_axis_off()
    figures.append(
        figure_row(
            "fig01_network_overview",
            save_figure(fig, "fig01_network_overview.png"),
            "Network overview",
            f"{rel(PATHS.processed_nodelist)}; {rel(PATHS.processed_edgelist)}",
            "Results: Network structure",
            timestamp,
        )
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    degree_counts = nodes["degree"].astype(int).value_counts().sort_index()
    ax.bar(degree_counts.index, degree_counts.values, color="#2563EB", edgecolor="white")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Node count")
    ax.set_title("Degree Distribution")
    ax.grid(axis="y", alpha=0.25)
    figures.append(
        figure_row(
            "fig02_degree_distribution",
            save_figure(fig, "fig02_degree_distribution.png"),
            "Degree distribution",
            rel(PATHS.processed_nodelist),
            "Results: Network structure",
            timestamp,
        )
    )

    benchmark = pd.read_csv(PATHS.metrics / "benchmark_table.csv").sort_values("test_auc_mean")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    y = np.arange(len(benchmark))
    ax.errorbar(
        benchmark["test_auc_mean"],
        y,
        xerr=benchmark["test_auc_std"],
        fmt="o",
        color="#2563EB",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(benchmark["model"])
    ax.set_xlabel("Test AUC")
    ax.set_title("Model Comparison")
    ax.grid(axis="x", alpha=0.25)
    figures.append(
        figure_row(
            "fig03_model_comparison",
            save_figure(fig, "fig03_model_comparison.png"),
            "Model comparison",
            rel(PATHS.metrics / "benchmark_table.csv"),
            "Results: Model benchmark",
            timestamp,
        )
    )

    ablation = pd.read_csv(PATHS.metrics / "ablation_table.csv").sort_values("test_auc_mean")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    y = np.arange(len(ablation))
    ax.errorbar(ablation["test_auc_mean"], y, xerr=ablation["test_auc_std"], fmt="o", color="#D97706", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ablation["feature_set"])
    ax.set_xlabel("Test AUC")
    ax.set_title("Ablation Results")
    ax.grid(axis="x", alpha=0.25)
    figures.append(
        figure_row(
            "fig04_ablation_results",
            save_figure(fig, "fig04_ablation_results.png"),
            "Ablation results",
            rel(PATHS.metrics / "ablation_table.csv"),
            "Results: Ablation",
            timestamp,
        )
    )

    budget_curve = pd.read_csv(PATHS.intervention / "phase05_budget_curve.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for strategy, group in budget_curve.groupby("strategy", sort=True):
        ax.plot(
            group["budget_k_pct"],
            group["reduction_vs_baseline_pct"],
            marker="o",
            linewidth=1.8,
            color=STYLE_COLORS.get(strategy, "#111827"),
            label=strategy.upper(),
        )
    ax.set_xlabel("Budget (% of nodes)")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.set_title("Budget Curve")
    ax.grid(alpha=0.25)
    ax.legend(title="Strategy")
    figures.append(
        figure_row(
            "fig05_budget_curves",
            save_figure(fig, "fig05_budget_curves.png"),
            "Budget curves",
            rel(PATHS.intervention / "phase05_budget_curve.csv"),
            "Results: Budget curves",
            timestamp,
        )
    )

    uncertainty = pd.read_csv(PATHS.intervention / "phase05_uncertainty_summary.csv")
    uncertainty_plot = uncertainty[
        (uncertainty["metric_name"] == "reduction_vs_baseline_pct")
        & (uncertainty["budget_k_pct"].isin([1, 5, 10, 20]))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    offsets = {"random": -0.3, "degree": -0.1, "betweenness": 0.1, "gnn": 0.3}
    for strategy, group in uncertainty_plot.groupby("strategy", sort=True):
        x = group["budget_k_pct"].astype(float) + offsets.get(strategy, 0)
        y = group["point_estimate"].astype(float)
        yerr = np.vstack([y - group["lower_interval"], group["upper_interval"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=STYLE_COLORS.get(strategy), label=strategy.upper())
    ax.set_xlabel("Budget (% of nodes)")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.set_title("Uncertainty Intervals")
    ax.grid(alpha=0.25)
    ax.legend(title="Strategy")
    figures.append(
        figure_row(
            "fig06_uncertainty_plots",
            save_figure(fig, "fig06_uncertainty_plots.png"),
            "Uncertainty plots",
            rel(PATHS.intervention / "phase05_uncertainty_summary.csv"),
            "Results: Uncertainty",
            timestamp,
        )
    )

    parameter = pd.read_csv(PATHS.intervention / "phase05_parameter_sweep.csv")
    parameter_plot = parameter[parameter["budget_k_pct"] == 10].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for strategy, group in parameter_plot.groupby("strategy", sort=True):
        grouped = group.groupby("beta", sort=True)["mean_reduction_vs_baseline_pct"].mean().reset_index()
        ax.plot(grouped["beta"], grouped["mean_reduction_vs_baseline_pct"], marker="o", label=strategy.upper(), color=STYLE_COLORS.get(strategy))
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean SIR reduction (%)")
    ax.set_title("Parameter Sweep at 10% Budget")
    ax.grid(alpha=0.25)
    ax.legend(title="Strategy")
    figures.append(
        figure_row(
            "fig07_parameter_sweep",
            save_figure(fig, "fig07_parameter_sweep.png"),
            "Parameter sweep plots",
            rel(PATHS.intervention / "phase05_parameter_sweep.csv"),
            "Supplement: Parameter sensitivity",
            timestamp,
        )
    )

    final = pd.read_csv(PATHS.intervention / "final_comparison.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pivot = final.pivot(index="budget_k_pct", columns="strategy", values="reduction_vs_baseline_pct")
    x = np.arange(len(pivot.index))
    width = 0.18
    strategies = [s for s in ["random", "degree", "betweenness", "gnn"] if s in pivot.columns]
    for idx, strategy in enumerate(strategies):
        ax.bar(
            x + (idx - (len(strategies) - 1) / 2) * width,
            pivot[strategy],
            width=width,
            label=strategy.upper(),
            color=STYLE_COLORS.get(strategy),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}%" for v in pivot.index])
    ax.set_xlabel("Budget")
    ax.set_ylabel("SIR reduction (%)")
    ax.set_title("Scenario Reduction by Strategy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Strategy")
    figures.append(
        figure_row(
            "fig08_scenario_reduction",
            save_figure(fig, "fig08_scenario_reduction.png"),
            "Scenario reduction plots",
            rel(PATHS.intervention / "final_comparison.csv"),
            "Results: Intervention comparison",
            timestamp,
        )
    )

    timeseries = pd.read_csv(PATHS.intervention / "phase05_timeseries_results.csv")
    ts = timeseries[
        (timeseries["param_id"] == "beta0.25_gamma0.10")
        & (timeseries["budget_k_pct"].isin([0, 10]))
    ].copy()
    ts_plot = ts.groupby(["strategy", "time_step"], sort=True)["mean_ever_infected"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for strategy, group in ts_plot.groupby("strategy", sort=True):
        label = "NO INTERVENTION" if strategy == "baseline" else strategy.upper()
        ax.plot(
            group["time_step"],
            group["mean_ever_infected"],
            marker="o",
            linewidth=1.8,
            color=STYLE_COLORS.get(strategy, "#111827"),
            label=label,
        )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean ever infected per household")
    ax.set_title("Saved Time-Series Output")
    ax.grid(alpha=0.25)
    ax.legend(title="Strategy")
    figures.append(
        figure_row(
            "fig09_time_series",
            save_figure(fig, "fig09_time_series.png"),
            "Time-series plots",
            rel(PATHS.intervention / "phase05_timeseries_results.csv"),
            "Results: Scenario dynamics",
            timestamp,
        )
    )

    manifest = pd.DataFrame(figures)
    manifest.to_csv(FIGURE_DIR / "figure_manifest.csv", index=False)
    return figures


def figure_row(
    figure_id: str,
    filename: str,
    title: str,
    source_artifact: str,
    manuscript_section: str,
    timestamp: str,
) -> dict[str, str]:
    return {
        "figure_id": figure_id,
        "filename": filename,
        "title": title,
        "source_artifact": source_artifact,
        "source_script": SCRIPT,
        "generation_timestamp_utc": timestamp,
        "manuscript_section": manuscript_section,
    }


def write_traceability(figures: list[dict[str, str]], tables: list[dict[str, str]]) -> None:
    lines = [
        "# Artifact Traceability",
        "",
        "All figures and tables in the paper-ready package are generated from saved CSV/JSON artifacts.",
        "",
        "## Figures",
        "",
        "| Artifact | Source CSV/JSON | Generating script | Validation command | Manuscript destination |",
        "|---|---|---|---|---|",
    ]
    for fig in figures:
        lines.append(
            f"| `{fig['filename']}` | `{fig['source_artifact']}` | `{fig['source_script']}` | "
            "`python scripts/validate_paper_artifacts.py` | "
            f"{fig['manuscript_section']} |"
        )
    lines.extend(
        [
            "",
            "## Tables",
            "",
            "| Artifact | Source CSV/JSON | Generating script | Validation command | Manuscript destination |",
            "|---|---|---|---|---|",
        ]
    )
    for table in tables:
        lines.append(
            f"| `{table['csv_path']}` | `{table['source_artifact']}` | `{table['source_script']}` | "
            "`python scripts/validate_paper_artifacts.py` | "
            f"{table['manuscript_section']} |"
        )
    (DOC_DIR / "artifact_traceability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_consistency_report() -> None:
    checks = collect_consistency_checks()
    lines = [
        "# Paper Artifact Consistency Report",
        "",
        "This report was generated by `scripts/31_generate_paper_artifacts.py` from authoritative result artifacts. It classifies consistency findings without introducing new scientific claims.",
        "",
        "| Status | Area | Finding | Evidence | Action |",
        "|---|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['status']} | {check['area']} | {check['finding']} | {check['evidence']} | {check['action']} |"
        )
    (DOC_DIR / "consistency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_consistency_checks() -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    add = checks.append
    required_sources = [
        PATHS.eda_summary,
        PATHS.metrics / "basic_metrics.json",
        PATHS.metrics / "benchmark_table.csv",
        PATHS.metrics / "ablation_table.csv",
        PATHS.intervention / "final_comparison.csv",
        PATHS.intervention / "phase05_budget_curve.csv",
        PATHS.intervention / "phase05_uncertainty_summary.csv",
        PATHS.intervention / "phase05_parameter_sweep.csv",
        PATHS.intervention / "phase05_timeseries_results.csv",
    ]
    missing = [rel(path) for path in required_sources if not path.exists()]
    add(
        {
            "status": "fixed" if not missing else "unresolved",
            "area": "source artifacts",
            "finding": "Required paper-ready source artifacts are present.",
            "evidence": "none missing" if not missing else "; ".join(missing),
            "action": "Generated paper-ready package from these files.",
        }
    )

    readme = (PATHS.root / "README.md").read_text(encoding="utf-8")
    if "GNN dominates both evaluation layers" in readme:
        add(
            {
                "status": "warning",
                "area": "README",
                "finding": "README still contains an interpretive 10% budget sentence that predates PR2/PR3 cleanup.",
                "evidence": "`GNN dominates both evaluation layers`",
                "action": "Leave for manuscript/README narrative refresh; generated tables use authoritative values.",
            }
        )
    stale_filename_hits = []
    for path in [PATHS.root / "README.md", *list((PATHS.root / "reports").glob("*.tex"))]:
        if path.exists() and "sir_results.csv" in path.read_text(encoding="utf-8"):
            stale_filename_hits.append(rel(path))
    add(
        {
            "status": "fixed" if not stale_filename_hits else "unresolved",
            "area": "filenames",
            "finding": "Runtime README/report files should not reference retired SIR filename.",
            "evidence": "none found" if not stale_filename_hits else "; ".join(stale_filename_hits),
            "action": "No generated paper-ready artifact uses the retired SIR filename.",
        }
    )

    phase05_meta = json.loads((PATHS.intervention / "phase05_generation_metadata.json").read_text(encoding="utf-8"))
    status = "requires rerun" if phase05_meta.get("profile") == "smoke" else "fixed"
    add(
        {
            "status": status,
            "area": "Phase 05 profile",
            "finding": "Phase 05 artifacts are generated with the configured profile.",
            "evidence": f"profile={phase05_meta.get('profile')}, n_runs={phase05_meta.get('n_runs')}, t_max={phase05_meta.get('t_max')}",
            "action": "Run the paper profile before final manuscript claims." if status == "requires rerun" else "Paper profile is available.",
        }
    )
    return checks


def main() -> None:
    setup_dirs()
    apply_publication_style()
    timestamp = utc_now()
    tables = build_tables(timestamp)
    figures = build_figures(timestamp)
    write_traceability(figures, tables)
    write_consistency_report()
    print(f"Generated {len(figures)} figures and {len(tables)} table groups in {rel(PAPER_ROOT)}")


if __name__ == "__main__":
    main()
