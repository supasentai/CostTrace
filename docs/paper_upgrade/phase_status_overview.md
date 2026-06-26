# COSTTRACE Paper Upgrade - Phase Status Overview

Date: 2026-06-25  
Reviewer: Codex  
Scope: high-level repository review of the paper-upgrade phases, based on current local files and generated artifacts.

## Repository Review Summary

The repository contains a working COSTTRACE analysis pipeline, generated result artifacts, visualizations, a course-style report, a newly created research-manuscript TeX draft, and a complete phase roadmap under `docs/`. The paper-upgrade process has begun, but only Phase 00 has been formally executed as a phase-gated task.

## Phase Completion Matrix

| Phase | Focus | Current status | Evidence | Main blockers |
|---|---|---|---|---|
| Phase 00 | Submission scope and compliance gate | Partially completed | `docs/paper_upgrade/phase_00_submission_scope_gate.md` | Target journal, article type, author metadata, corresponding author, final title, and data-sharing permissions still require human confirmation. |
| Phase 01 | Manuscript identity, abstract, RQ, paper conversion | Not completed | `reports/costtrace_research_manuscript.tex` is a preliminary draft only | Abstract, bilingual title page, final author metadata, research-question integration, and full report-to-manuscript conversion are not acceptance-ready. |
| Phase 02 | Related work and citation base | Not completed | Placeholder in `reports/costtrace_research_manuscript.tex` | Literature review and verified citation library still need to be built. |
| Phase 03 | Dataset protocol and leakage audit | Completed with submission blockers | `docs/paper_upgrade/phase_03_dataset_protocol_and_leakage_control.md`; `results/audit/phase03_leakage_audit.json`; `reports/costtrace_research_manuscript.tex` | Raw-data redistribution permissions, original ethics details, official SASHTS citation, and multi-fold household validation remain unresolved. |
| Phase 04 | Modeling fixes, benchmarks, ablation | Not completed | Existing GraphSAGE artifacts under `results/model/` | Overfitting mitigation, multi-seed/fold evaluation, stronger baselines, and ablation are not completed. |
| Phase 05 | SIR sensitivity and budget evaluation | Not completed | Existing intervention artifacts under `results/intervention/` | Sensitivity analysis, uncertainty intervals, broader budget curve, and counterfactual assumption rewrite are not completed. |
| Phase 06 | Results narrative, figures, tables, consistency audit | Not completed | Figures and result CSV/JSON files exist | Paper-ready figure regeneration, generated manuscript tables, consistency audit, and final narrative are not completed. |
| Phase 07 | Reproducibility, tests, supplement | Not completed | Pipeline entrypoint `main.py`; requirements exist | Version-pinned environment, tests, CI/local validation commands, and supplement package are not completed. |
| Phase 08 | Language, references, cover letter, final submission gate | Not completed | Phase checklist exists under `docs/` | Final citation style, polished manuscript, cover letter, final formatting, and internal sign-off are not completed. |

## Work Completed So Far

1. Read the complete `docs/` Markdown roadmap and criteria files.
2. Created the Phase 00 gate document at `docs/paper_upgrade/phase_00_submission_scope_gate.md`.
3. Identified Journal of Thoracic Disease / AME as a provisional target based on local template and official author-guideline evidence.
4. Marked the journal, article type, reference style, title page, and data-sharing decisions as provisional where evidence is missing.
5. Created a preliminary LaTeX research-manuscript draft at `reports/costtrace_research_manuscript.tex`.

## Evidence Paths

- `docs/README_PHASE_ROADMAP.md`
- `docs/CRITERIA_MATRIX.md`
- `docs/CODEX_PROMPT_PACK.md`
- `docs/phase_00_submission_scope_gate.md`
- `docs/paper_upgrade/phase_00_submission_scope_gate.md`
- `reports/costtrace_research_manuscript.tex`
- `reports/report.tex`
- `README.md`
- `data/raw/SOURCE.md`
- `data/processed/eda_summary.json`
- `results/metrics/basic_metrics.json`
- `results/model/gnn_metrics.json`
- `results/intervention/final_comparison.csv`

## Validation Notes

- The TeX manuscript references existing figure artifacts: `visualizations/network_overview.png`, `visualizations/chart_heatmap.png`, and `visualizations/chart_reduction_by_strategy.png`.
- PDF compilation was not validated because no `xelatex`, `pdflatex`, or `tectonic` executable was available in PATH during the previous check.
- The current phase status should be interpreted as workflow status, not scientific validation of the manuscript.

## Remaining Risks

- The current manuscript draft is not submission-ready.
- JTD/AME target is provisional until confirmed by the corresponding author.
- The current GraphSAGE evaluation still needs leakage audit and stronger validation before publication claims.
- Raw data and report files may have licensing/privacy constraints and should not be added to a public release without confirmation.
