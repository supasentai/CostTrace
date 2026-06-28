# Phase 05 Scenario-Evaluation Artifacts

## Purpose

PR 3 adds reproducible Phase 05 scenario artifacts without training models, changing model outputs, or rewriting manuscript claims. The new artifacts preserve run-level and time-series detail so later figures can be generated from saved outputs instead of reconstructed aggregate summaries.

## Regeneration Commands

Smoke/default profile:

```bash
python scripts/30_phase05_scenario_artifacts.py --profile smoke
python scripts/validate_phase05_artifacts.py
```

Paper-grade profile, intentionally not run in PR 3 validation:

```bash
python scripts/30_phase05_scenario_artifacts.py --profile paper
python scripts/validate_phase05_artifacts.py
```

## Parameter Settings

| Profile | Runs | Time horizon | Beta grid | Gamma grid | Budgets | Strategies |
|---|---:|---:|---|---|---|---|
| `smoke` | 3 | 10 | 0.20, 0.25, 0.30 | 0.10 | 1% through 20% | random, degree, betweenness, gnn |
| `paper` | 50 | 30 | 0.15, 0.20, 0.25, 0.30, 0.35 | 0.05, 0.10, 0.15 | 1% through 20% | random, degree, betweenness, gnn |

The smoke profile is a validation artifact set, not a paper-grade estimate. The paper profile is configured for later full regeneration.

## Files Generated

| File | Role | Status |
|---|---|---|
| `results/intervention/phase05_run_level_results.csv` | Raw run-level scenario outcomes with strategy, budget, parameter setting, run id, seed, selected nodes, counterfactual metrics, and SIR reduction metrics. | Smoke-test artifact |
| `results/intervention/phase05_timeseries_results.csv` | True saved time-series output by strategy, budget, parameter setting, run id, seed, and time step. | Smoke-test artifact |
| `results/intervention/phase05_baseline_summary.csv` | Baseline SIR summary by parameter setting. | Smoke-test artifact |
| `results/intervention/phase05_parameter_sweep.csv` | Parameter-sweep summary with means, standard deviations, percentile intervals, and run counts. | Smoke-test artifact |
| `results/intervention/phase05_budget_curve.csv` | Budget curve for the default beta/gamma setting, including marginal SIR gain. | Smoke-test artifact |
| `results/intervention/phase05_budget_decision_table.csv` | Best strategy per budget by prevention, SIR reduction, and transmission coverage. | Smoke-test artifact |
| `results/intervention/phase05_uncertainty_summary.csv` | Percentile intervals across Phase 05 runs for key metrics. | Smoke-test artifact |
| `results/intervention/phase05_generation_metadata.json` | Generation profile and parameter metadata. | Smoke-test artifact |
| `results/figures/phase05/phase05_timeseries_curve.png` | Draft time-series curve generated from `phase05_timeseries_results.csv`. | Draft figure |
| `results/figures/phase05/phase05_parameter_sweep.png` | Draft parameter-sweep line plot generated from `phase05_parameter_sweep.csv`. | Draft figure |
| `results/figures/phase05/phase05_budget_curve.png` | Draft budget curve generated from `phase05_budget_curve.csv`. | Draft figure |
| `results/figures/phase05/phase05_uncertainty_intervals.png` | Draft uncertainty plot generated from `phase05_uncertainty_summary.csv`. | Draft figure |

## Known Limitations

- The smoke artifacts use only 3 runs and a shortened time horizon, so they are validation outputs rather than manuscript-ready estimates.
- The generator reuses the current static household SIR mechanics and current saved GNN risk scores; it does not introduce retraining or a new model.
- Random strategy uncertainty reflects the smoke run count only.
- Deterministic strategies have stochastic SIR variation but fixed selected nodes for a given budget and score column.
- Later paper-grade execution should run the `paper` profile and then refresh any paper/report figures from these saved artifacts.

