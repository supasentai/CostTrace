# Artifact Traceability

All figures and tables in the paper-ready package are generated from saved CSV/JSON artifacts.

## Figures

| Artifact | Source CSV/JSON | Generating script | Validation command | Manuscript destination |
|---|---|---|---|---|
| `results/paper_ready/figures/fig01_network_overview.png` | `data/processed/sashts/nodelist.csv; data/processed/sashts/edgelist.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Network structure |
| `results/paper_ready/figures/fig02_degree_distribution.png` | `data/processed/sashts/nodelist.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Network structure |
| `results/paper_ready/figures/fig03_model_comparison.png` | `results/metrics/benchmark_table.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Model benchmark |
| `results/paper_ready/figures/fig04_ablation_results.png` | `results/metrics/ablation_table.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Ablation |
| `results/paper_ready/figures/fig05_budget_curves.png` | `results/intervention/phase05_budget_curve.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Budget curves |
| `results/paper_ready/figures/fig06_uncertainty_plots.png` | `results/intervention/phase05_uncertainty_summary.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Uncertainty |
| `results/paper_ready/figures/fig07_parameter_sweep.png` | `results/intervention/phase05_parameter_sweep.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Supplement: Parameter sensitivity |
| `results/paper_ready/figures/fig08_scenario_reduction.png` | `results/intervention/final_comparison.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Intervention comparison |
| `results/paper_ready/figures/fig09_time_series.png` | `results/intervention/phase05_timeseries_results.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Scenario dynamics |

## Tables

| Artifact | Source CSV/JSON | Generating script | Validation command | Manuscript destination |
|---|---|---|---|---|
| `results/paper_ready/tables/table01_dataset_summary.csv` | `data/processed/sashts/eda_summary.json` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Methods: Dataset |
| `results/paper_ready/tables/table02_graph_summary.csv` | `results/metrics/basic_metrics.json` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Network structure |
| `results/paper_ready/tables/table03_model_benchmark.csv` | `results/metrics/benchmark_table.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Model benchmark |
| `results/paper_ready/tables/table04_ablation.csv` | `results/metrics/ablation_table.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Ablation |
| `results/paper_ready/tables/table05_strategy_comparison.csv` | `results/intervention/final_comparison.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Intervention comparison |
| `results/paper_ready/tables/table06_budget_analysis.csv` | `results/intervention/phase05_budget_decision_table.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Budget curves |
| `results/paper_ready/tables/table07_uncertainty_summary.csv` | `results/intervention/phase05_uncertainty_summary.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Results: Uncertainty |
| `results/paper_ready/tables/table08_parameter_sensitivity.csv` | `results/intervention/phase05_parameter_sweep.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Supplement: Parameter sensitivity |
| `results/paper_ready/tables/table09_supplementary_full_metrics.csv` | `results/model/full_pytorch_graphsage_full_metrics.csv; results/metrics/benchmark_full_metrics.csv; results/metrics/ablation_full_metrics.csv` | `scripts/31_generate_paper_artifacts.py` | `python scripts/validate_paper_artifacts.py` | Supplement: Full metrics |
