# Metric Consistency Issues

This table records retired values and stale artifact names found during PR 2. "Fixed now" means the change was direct and source-backed by an authoritative CSV/JSON artifact or by an existing current filename. "TODO" means the reference should be handled in a later paper/report refresh, usually because it is generated notebook output, an audit checklist item, or interpretation text that should not be rewritten in this PR.

| File path | Current stale text/value | Authoritative artifact path | Recommended replacement | Severity | Fixed now or TODO |
|---|---|---|---|---|---|
| `README.md` | Key finding `test AUC = 0.7669` and `test Average Precision = 0.8995` | `results/model/full_pytorch_graphsage_summary.csv` | `test AUC = 0.5772 +/- 0.0443`; `test Average Precision = 0.7679 +/- 0.0246`; 15 runs | HIGH | Fixed now |
| `README.md` | Key finding GNN prevention `26.8%`/`49.0%`, SIR reductions `22.2%`/`42.7%` | `results/intervention/final_comparison.csv` | GNN prevention `17.6%`/`28.1%`; SIR reductions `12.6%`/`22.7%` at 5%/10% budgets | HIGH | Fixed now |
| `README.md` | Model performance Test row `0.7669`, `0.8995` plus single-split F1/precision/recall | `results/model/full_pytorch_graphsage_summary.csv` | Replace section with current 15-run summary table in a later README refresh | MEDIUM | TODO |
| `README.md` | Final Strategy Comparison 5% GNN row `14.3%`, `26.8%`, `22.2%` | `results/intervention/final_comparison.csv` | Transmission coverage `11.5%`, prevention `17.6%`, SIR reduction `12.6%` | HIGH | Fixed now |
| `README.md` | Final Strategy Comparison 10% GNN row `100.0%`, `26.2%`, `49.0%`, `42.7%` | `results/intervention/final_comparison.csv` | Precision `94.1%`, transmission coverage `19.9%`, prevention `28.1%`, SIR reduction `22.7%` | HIGH | Fixed now |
| `README.md` | 5% interpretation says GNN reaches `26.8%` prevention | `results/intervention/final_comparison.csv` | `17.6%` prevention | MEDIUM | Fixed now |
| `README.md` | 10% interpretation says GNN dominates, blocks 75 edges, reaches `49.0%`, SIR `42.7%` | `results/intervention/final_comparison.csv`; `results/intervention/final_strategy_summary.json` | Later rewrite should state current 10% comparison carefully: Random is best by prevention, GNN has `28.1%` prevention and `22.7%` SIR reduction | REVIEWER-CRITICAL | TODO |
| `docs/CRITERIA_MATRIX.md` | Checklist item cites `0.7669`, `0.8995`, `26.8%/49.0%` | Current model and intervention artifacts | Replace only when refreshing the audit checklist; current values are documented in `authoritative_artifacts.md` | MEDIUM | TODO |
| `docs/CRITERIA_MATRIX.md` | Coordination note cites train AUC `0.9914` vs test AUC `0.7669` | `results/model/full_pytorch_graphsage_summary.csv` | Refresh modeling note from current Phase 04 artifact set | MEDIUM | TODO |
| `docs/phase_01_manuscript_identity_and_framing.md` | Phase note cites `0.7669`, `0.8995`, `26.8%/49.0%` | Current model and intervention artifacts | Refresh during manuscript identity/framing pass | MEDIUM | TODO |
| `docs/phase_04_modeling_benchmarks_and_ablation.md` | Phase note cites test AUC `0.7669` | `results/model/full_pytorch_graphsage_summary.csv` | Refresh during Phase 04 doc cleanup | MEDIUM | TODO |
| `notebooks/intervention.ipynb` | Generated output cells contain retired `26.8`, `49.0`, `22.2`, `42.7` | `results/intervention/final_comparison.csv` | Re-run or clear notebook outputs in a notebook-focused PR | MEDIUM | TODO |
| `reports/costtrace_research_manuscript.tex` | Phase 04 GraphSAGE values `0.5842 +/- 0.0487` and `0.7809 +/- 0.0295` | `results/model/full_pytorch_graphsage_summary.csv` | `0.5772 +/- 0.0443` and `0.7679 +/- 0.0246` | HIGH | Fixed now |
| `reports/report.tex` | Artifact name `sir_results.csv` | `results/intervention/sir_intervention_results.csv` | `sir_intervention_results.csv` | HIGH | Fixed now |
| `docs/CRITERIA_MATRIX.md`; `docs/phase_06_results_figures_tables_and_narrative.md` | Checklist text mentions `sir_results.csv` as a known mismatch | `results/intervention/sir_intervention_results.csv` | Leave as historical checklist context until phase docs are refreshed | LOW | TODO |
| `results/intervention/final_comparison.csv`; `results/intervention/counterfactual_results.csv` | Search hits for `49.0` and `22.2` | Same files | No replacement; these are valid current 10% Degree values, not stale GNN values | LOW | No change |
| `visualizations/network_overview.svg`; `results/orders/trace_candidate_features.csv`; `results/model/full_pytorch_graphsage_full_metrics.csv` | Incidental numeric matches from geometry or generated data rows | Generated artifacts | No replacement; not paper prose references | LOW | No change |

