# Standalone snapshot and changes

Source branch: `thesis_hybrid_model`. Source HEAD at preparation: `5c3f4f3f17715f5cd81fbfb72ecf752bf528bad5`. The snapshot also includes the existing uncommitted experiment/training edits and the two newly supplied helper files. It intentionally reflects the current working tree rather than only the committed HEAD.

The original checkout, branch, index and remotes were not edited. All organization and path changes took place in an independent directory without `.git`. Publishing using the provided instructions creates a fresh root commit with no relationship to the original commit graph.

## Organization changes

| Source location | Standalone location | Reason |
|---|---|---|
| Root `conservative_pilot_utils.jl` | `thesis_experiments/shared/conservative_pilot_utils.jl` | Shared dataset loading and optional source pretraining belong beside experiment orchestration |
| Root `runtime.jl` | `thesis_experiments/shared/reviewer_followup_runtime.jl` | Give the follow-up runtime an explicit name beside the other shared helper |
| Four result-figure builder modules from `thesis/analysis/report_figures/` | `analysis/` | Preserve numerical result and paired-event figure reproduction without including the thesis directory |
| Relevant source-to-target evaluation tables under `thesis/analysis/figures_source_controls_target/` | `results/thesis/source_target_evaluation/` | Preserve the exact evidence used by the no-domain-shift table |
| Combined historical reviewer case index | `results/thesis/reviewer_followups_initialfix_v1/indices/reported_cases.tsv` | Retain only two reported cases and remove obsolete case names from the file name |

All runner includes and source-fingerprint path lists were updated for the two relocated helpers. The conservative helper remains two levels below the repository root, preserving its existing `@__DIR__/../..` path logic. Scientific model/training code, experiment settings, data bytes and checkpoint bytes were not changed during preparation.

Figure builders now write to `docs/figures/`; they no longer read thesis TeX files or require obsolete endpoint diagnostics. The retained entry point generates only the seven numerical figures referenced by the active results chapter and appendix. Historical text run indices have portable repository-relative directory fields. The filtered pretraining summary points to the included selected checkpoint.

## Retained and excluded material

Retained: five current runners, two restored shared helpers, rank-selection implementation, the eight Julia source files they load, the Julia project and manifest, five original required ensembles, generation settings, the selected pretrained checkpoint, thesis-referenced saved evidence, saved-data plotting dependencies, and verification/documentation files.

Excluded: the entire `thesis/` directory, original `.git`, unrelated history and branches, `tools/`, `tmp/`, `.local/`, old notebooks, numerical-generation environments/scripts not used by the retained prepared-data runners, unused model/plotting helpers, unused datasets and pretrained models, old/exploratory experiments, editorial backups, caches, and result campaigns absent from the active thesis. See [RESULTS_PROVENANCE.md](RESULTS_PROVENANCE.md) for the exact campaign and subcase selection.

The full local selection audit and original working-tree status snapshot are delivered beside this clean directory for review and are not part of the public project. The public `file_checksums.sha256` lists the exact prepared files, excluding itself and ignored regenerated figures/caches.

The public content is approximately 723 MB, dominated by approximately 522 MB of necessary original input ensembles and 200 MB of selected result evidence. Four ensembles exceed the ordinary-Git 100 MiB file limit. All JLD2/BSON objects are configured for Git LFS. Fresh results, generated figures and caches are ignored.

## Verification boundaries

The saved historical results are retained as evidence. Fresh execution uses the current code and creates new fingerprints; this migration does not assert that every historical result can be regenerated bit for bit after earlier scientific-code edits or on different hardware. Full multi-day training campaigns were not run as part of packaging. The exact checks and their outcomes are recorded in [VERIFICATION.md](VERIFICATION.md).
