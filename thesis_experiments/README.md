# Thesis experiment guide

For environment setup, input files, shared helper locations and execution
commands, see the [project README](../README.md). This folder contains the reported
experiments; the commented scripts expose all scientific settings directly.

## Runner map

| Script | Thesis experiment | Output subfolder |
|---|---|---|
| [00_source_pretraining.jl](00_source_pretraining.jl) | Optional flux-correction source pretraining, Appendix A.1 | `source_pretraining/` |
| [01_main_experiments.jl](01_main_experiments.jl) | Experiments 1-4: adaptation parameterisation, temporal training, gauge geometry/drag profile, observation noise | `main/` |
| [02_gauge_and_curriculum_followups.jl](02_gauge_and_curriculum_followups.jl) | Two-gauge comparison and shortened curriculum | `followups/` |
| [03_no_domain_shift_control.jl](03_no_domain_shift_control.jl) | Sparse source-domain long-128 fine-tuning, tested on source and target events | `source_control/` |
| [04_profile_adaptation.jl](04_profile_adaptation.jl) | CPU profiling of six saved adaptation models | `adaptation_profile/` |

[experiment_io.jl](experiment_io.jl) loads the shared implementation and preserves
the original timed checkpoint/result format; it is not a standalone runner.

Runner 01 keeps the four main experiments together because each fold's
validation-selected method/objective feeds the next experiment. Its baseline
settings are five outer folds, three repeats, two training events, four validation
events, 1,024 training transitions, observations every 16 steps and 200 epochs.
LoRA rank selection uses only the outer-training pool and runs before the final fits.

All target fits start from the same five-layer, 2,706-parameter flux-correction
checkpoint. Supervision uses water level only; evaluation compares full clean
water-level and velocity fields over continuous 72-hour forecasts. Repeat MSEs are
averaged within each event before equally weighting events and taking the square
root for RMSE. The main reported intervals are 0-12 and 48-72 hours.

For the 127-detached-step comparison, the historical output name `one_step` is
retained. The current configuration uses endpoint 128, `detach_gradients=true`
and `gradient_steps=1`: only the final transition is differentiated. This replaces
the historical cached-predecessor option; identical historical numerical
trajectories or timings are not assumed.

Runner 02 retains each fold's main temporal choice for the two-gauge comparison.
The shortened curriculum instead uses 10/10/10/170 epochs at lead sets
`[16]`, `[16,32]`, `[16,32,64]` and `[16,32,64,128]`.
To use existing completed main fits for runners 02/04, set `baseline_directory`
to `results/thesis/thesis_flux_correction_v4` in those scripts.

## Saved results

All paths below are relative to `results/thesis_reproduction/`.

- Main fits use `main/runs/<method>__<objective>__<profile>__<layout>__noise<N>cm/fold_<f>/repeat_<r>/`.
  Source-control fits use `source_control/runs/long128/fold_<f>/repeat_<r>/`.
- Each fit's `result.jld2` has key `result`: configuration, split, history,
  validation-selected epoch/loss, epoch/update counts, parameter count, loss scales
  and timings. `checkpoint.jld2` has key `saved`: resumable training state and
  accumulated times. Follow-up checkpoints also retain configuration/split metadata.
- `best_model.bson` contains validation-selected weights; `final_model.bson` contains
  the last training iterate. `training_history.tsv` is the readable learning curve.
  New histories retain raw gradient and parameter-update norms, without a clipping-fraction column.
- Main/source `diagnostics/report_metrics.jld2` has key `result`, containing
  `series`, `summaries` and `times`. Companion TSVs retain per-event errors.
  Main `source_retention/` reports use the matched source events for 12 hours.
- `main/manifest.jld2` records input/code fingerprints and common settings.
  `main/selections/` holds method/temporal validation tables. `main/rank_selection/`
  holds candidate spectra/selections and `selected_ranks.toml`. `run_index.tsv`,
  `event_metrics.tsv` and `summary.tsv` retain the original main-analysis columns.
- Follow-up fits are under `followups/runs/<case>/fold_<f>/repeat_<r>/`.
  Their `evaluation/interval_metrics.jld2` has key `rows`. Matched baseline scores
  are under `followups/references/`; aggregate tables are under
  `followups/partial_reports/gauge_one-curriculum_fast_f1-2-3-4-5_r1-2-3/`.
- Source-control event averages and summaries are saved beside its run index.
  `source_control/target_evaluation/event_metrics.tsv` scores the source-fine-tuned
  models on target events. Compare with runner 01's stage-2 `long` target-trained
  scores for the no-domain-shift table.
- Profiling saves `samples.tsv`, `summary.tsv`, per-method call-stack text files and
  `environment.txt`, using three warmups and ten samples. Timings depend on the runtime/hardware.
- Optional pretraining saves under `source_pretraining/pretraining/<specification hash>/flux_correction/`:
  `complete.jld2` key `result`, `checkpoint.jld2` key `state`, and
  `specification.jld2` keys `specification` and `test_loss`, plus model/history files.
  Its summary table is in the parent hash directory. Training uses source events
  1-4, validation 29-34 and test 35-40 for 500 epochs.

Use a new output directory for changed settings rather than overwriting historical
runs. Keep JLD2 keys, event/repeat records and metric units when analysing results.
The transferred target data do not record solver tolerances; exact data provenance
cannot be reconstructed from those files alone.

