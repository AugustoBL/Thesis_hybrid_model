# Experiment guide

Run from the repository root with the Starting_networks Julia environment:

```powershell
julia --project=Starting_networks thesis_experiments/01_sparse_adaptation_experiments.jl
```

Each runner executes directly. Including a runner loads its definitions and dependencies
without calling `main()`. Include runners in separate modules or Julia sessions because
they define different settings under the same constant names.

| Runner | Experiment | Output under `results/experiments/` |
|---|---|---|
| `00_pretrain_flux_correction.jl` | Source pretraining | `source_pretraining/` |
| `01_sparse_adaptation_experiments.jl` | Adaptation methods, training objectives, gauge layouts/drag, observation noise | `sparse_adaptation/` |
| `02_gauge_density_curriculum.jl` | Two gauges and a faster curriculum | `gauge_curriculum/` |
| `03_source_domain_control.jl` | Source-domain adaptation and transfer evaluation | `source_domain_control/`, `source_to_target/` |
| `04_profile_adaptation_cost.jl` | Isolated operation costs for six saved adaptation models | `adaptation_profiling/` |

Inputs remain in `Data/` and `Trained_models/`. Existing `results/reported/` files
are preserved. Runners 02 and 04 use runner 01's output by default; set their
`baseline_directory` to `results/reported/sparse_adaptation` to analyze existing fits.
The unused control dataset is no longer loaded by runner 01.

`experiment_dependencies.jl` loads `Surge`, diagnostics, and source-pretraining persistence.
Reusable data preparation, supervised and sparse training, forecasting, adaptation
checkpointing, and event-weighted aggregation live in `src/`.
`shared/source_pretraining.jl` contains only source checkpoint/model persistence.
`shared/gauge_curriculum_analysis.jl` selects the retained report columns and compares
paired conditions.

The numerical settings remain five outer folds, three repeats, two training events,
four validation events, 1,024 training transitions, observations every 16 steps,
and 200 adaptation epochs. LoRA ranks are selected within the outer-training pool.
Source pretraining uses events 1-4, validation 29-34, test 35-40, and 500 epochs.
Its supervised noise and learning-rate schedule differ from sparse adaptation and
are retained in a separate `train_model!` method.

Forecast metrics retain the original Float64 reductions, spatial masks, interval
boundaries, and bias/anomaly definitions. Aggregation averages repeat MSE within
events before weighting events equally; RMSE is calculated afterward. The output
label `one_step` remains because existing results identify the 127-detached-step,
lead-128 comparison by that name. The unused `native_one_step` setting is removed.

The gauge label `gauge_one` is now `two_gauges` (stations 2 and 99).
The comparison conditions are `four_gauges`, `eight_gauges`, `balanced`, and
`curriculum`, corresponding to the stored result labels `gauges_two`, `gauges_four`,
`multi128`, and `curriculum_original`. Existing result files retain their labels.
Baseline scores are read directly from runner 01's diagnostic TSVs; they are not
forecast again or saved as duplicate reference files.

## Retained outputs

- Adaptation runs: `result.jld2`, `best_model.bson`, `training_history.tsv`, and
  resumable `checkpoint.jld2` while fitting. The checkpoint's `saved` key retains
  configuration, split, preparation time, and trainer state (model, optimizer, RNG,
  best parameters, history, and accumulated training time).
- Main campaign: `run_index.tsv`, method/temporal selection tables, rank selection
  spectra/decisions/TOML files, per-run `diagnostics/metrics.tsv`, `event_metrics.tsv`,
  and `summary.tsv`. Per-step TSV series are retained only for full adaptation,
  uniform drag, the four end gauges, no noise, and balanced/long/one_step objectives,
  matching the stored series. The rank pilot retains its final iterate as an
  intermediate model because update spectra must use that iterate.
- Gauge/curriculum: the existing interval JLD2/TSV evaluation files,
  `indices/reported_cases.tsv`, and event, summary, and comparison tables at the
  campaign root. No selection-specific report subdirectories are created.
- Source control: existing run, diagnostic, event, and summary tables. The separate
  `source_to_target/` campaign retains `event_metrics.tsv` and `summary.csv`.
- Source pretraining: `pretraining/flux_correction/complete.jld2`, `history.tsv`,
  `specification.jld2`, the selected model and resumable checkpoint, with
  `pretraining/summary.tsv`. Output paths no longer contain a specification hash.
- Profiling: `samples.tsv`, `summary.tsv`, and `environment.txt`; three warmups,
  ten samples, and the original sampling seed. Isolated phase timings also measure
  allocations/GC, standalone forward evaluation, and optimizer-only costs. These
  differ from cumulative timings inside `train_model!`. Sampling-stack text reports
  are removed. Preparation time remains outside the trainer; total training time
  and training phase times are measured inside it.

Fingerprint manifests, source-retention model copies/reports, binary diagnostic
archives, per-step series outside the retained cases, and extra final-model exports
are no longer produced by the experiment runners. Metrics represented in the stored
results remain available. No existing results are changed or migrated.

Use a new output directory when changing settings. Saved adaptation configurations
and splits are checked before reuse or resume; finite-loss/gradient/forecast and
paired-event safeguards remain. Hash-based checks and naming compatibility aliases
are removed. Exact historical timings and full numerical equivalence require reruns;
no full campaigns are needed for loading or lightweight smoke checks.
