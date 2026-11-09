# Experiment guide

These experiments test how a pretrained surge model adapts to sparse water-level
observations. They compare adaptation methods, forecast leads, gauge layouts,
hidden drag and observation noise. Separate scripts test source-domain adaptation,
a faster curriculum and operation costs.

## Setup and run order

Use the Julia environment at the repository root. See the
[root README](../README.md) for package setup. Run the commands below from that root.
No `--run` argument is needed. Each script calls `main()` only when run directly.
Use a separate Julia session for each script.

| Script | Purpose | Required inputs |
|---|---|---|
| `00_pretrain_flux_correction.jl` | Train the source flux-correction CNN | `Data/24_hour_source.jld2` |
| `01_sparse_adaptation_experiments.jl` | Compare methods, temporal objectives, geometry and noise | Pretrained model and `uniform`, `interior` and `boundary` datasets |
| `02_gauge_density_curriculum.jl` | Test two gauges and a faster curriculum | Pretrained model, `uniform` data and matching saved results from 01 |
| `03_source_domain_control.jl` | Adapt without hidden drag and test transfer to uniform drag | Pretrained model and `control` and `uniform` datasets |
| `04_profile_adaptation_cost.jl` | Measure separate operations for six saved models | Pretrained model, `uniform` data and six balanced fits from 01 |

The default pretrained model is
`Trained_models/flux_correction_pretrained.bson`.
The adaptation datasets are `Data/thesis_v4/<profile>.jld2`.

1. Use the existing pretrained model or run 00 to make a new one.
2. Run 01 before 02 and 04.
3. Run 03 independently of 01.

Runner 00 saves its model under its output directory. It does not replace
`Trained_models/flux_correction_pretrained.bson`. To use a new model, set
`pretrained_file` in the later runners to its `best_model.bson` path.

Runners 02 and 04 read `results/experiments/sparse_adaptation` by default.
Their `baseline_directory` can instead point to
`results/reported/sparse_adaptation` to use stored fits. Runner 02 needs
temporal selection tables and full-adaptation four/eight-gauge, balanced
and curriculum diagnostics for matching folds and repeats. Check that these
fits exist in the chosen baseline directory. Runner 04 needs all six methods
with the balanced objective, uniform drag, layout `both`, no noise, fold 1
and repeat 1.

## Shared adaptation settings

Settings are near the top of each script. The runners have no command-line
configuration interface. Change constants and configuration tuples in the
script when making a new experiment.

| Setting | Default in 01, 02 and 03 |
|---|---|
| Events | 18 events split into five test folds |
| Training draws | Three repeats per fold with two training events |
| Validation | Four fixed validation events per fold |
| Seeds for splits and training | `selection_seed=42000` and `training_seed=420000` |
| Training data | First 1,024 transitions with observations every 16 steps |
| Training | 200 epochs and batches of four windows |
| Window stride and available leads | Stride 128 and leads 16, 32, 64 and 128 |
| Optimizer | Adam with learning rate `3e-6`, betas `(0.9, 0.999)` and epsilon `1e-8` |
| Validation interval | Every five epochs with early stopping disabled |
| Initial-state cache | Refreshed every 25 epochs |
| Evaluation | Up to 72 hours with intervals `(0, 12]`, `(12, 24]`, `(24, 48]` and `(48, 72]` hours |

Leads, observation spacing and window stride are in model steps, not hours.
Sparse training observes water levels only. Forecast diagnostics also evaluate
velocity and use cells 9:92 for interior errors.
Repeats are averaged within each event before events receive equal weight.
RMSE is calculated after averaging squared errors.

Check `data_directory`, `pretrained_file`, `output_directory` and
`baseline_directory` where present. Runner 03 defines its separate transfer output
path inside `main()`. Use new output directories when changing inputs or settings.
Keep split seeds and noise seeds fixed when comparing conditions.

## 00: source pretraining

```powershell
julia --project=. thesis_experiments/00_pretrain_flux_correction.jl
```

This trains a conservative CNN with a learned flux correction. It uses source
events 1:4 for training, 29:34 for validation and 35:40 for testing.
The network has 16 hidden channels and filter width 3. Training uses 500 epochs,
batch size 50, initial learning rate `1e-4`, input noise `0.001f0` and seed 2026.
The learning-rate schedule and checkpoint interval are in `pretraining_settings`.

Outputs go to `results/experiments/source_pretraining/`:

- `pretraining/flux_correction/best_model.bson`: best validation model.
- `complete.jld2`: completed-fit information in the same model directory.
- `history.tsv` and `checkpoint.jld2`: losses and progress in that directory.
- `specification.jld2`: split, architecture, settings and test loss in that directory.
- `pretraining/summary.tsv`: validation and test MSE, best epoch and parameter count.

## 01: sparse adaptation

```powershell
julia --project=. thesis_experiments/01_sparse_adaptation_experiments.jl
```

The script first selects LoRA ranks within each outer training pool. It uses a
10-epoch full-adaptation pilot and candidates trained for up to 50 epochs.
Energy thresholds are 0.8, 0.9, 0.95 and 0.99. Saved fold ranks are reused when
`rank_selection/selected_ranks.toml` exists.

Each outer fold then runs four stages:

1. Compare `full`, `output`, `last_two`, `first_two`, `lora` and `film` with
   the balanced objective. Select the method by mean validation loss across repeats.
2. Compare temporal objectives with the selected method. Select an objective
   for the next stages. Also run `fixed32`, `fixed64` and `one_step` controls.
3. Vary drag profile and gauge layout with the selected method and objective.
4. Test observation noise of 0.01, 0.05, 0.10 and 0.20 m with uniform drag
   and four gauges.

| Objective | Meaning |
|---|---|
| `short`, `fixed32`, `fixed64`, `long` | Endpoint loss at lead 16, 32, 64 or 128 |
| `balanced` | Loss across leads 16, 32, 64 and 128 |
| `curriculum` | Add longer leads at fractions 0.25, 0.375 and 0.5 of training |
| `detached` | Forecast to lead 128 with gradients through the last 16 steps |
| `one_step` | Forecast to lead 128 with gradients through only the last step |

| Layout | Water-level cell indices |
|---|---|
| `both` | 2, 4, 97, 99 |
| `one` | 2, 4, 6, 8 |
| `eight` | 2, 4, 6, 8, 93, 95, 97, 99 |
| `spread` | 10, 37, 64, 91 |

`geometry_cases` lists the tested profile/layout combinations. The script does
not test every possible combination.

Outputs go to `results/experiments/sparse_adaptation/`:

- `run_index.tsv`: stages, conditions, fit paths, validation scores and timing.
- `selections/method_fold_<fold>.tsv` and `temporal_fold_<fold>.tsv`: sorted validation scores.
- `rank_selection/`: pilot and candidate fits, update spectra and selected ranks.
- `event_metrics.tsv`: interval scores by stage, condition, model and event.
- `summary.tsv`: event-averaged errors and training costs for each condition.

Fits are stored under
`runs/<method>__<objective>__<profile>__<layout>__noise<N>cm/fold_<fold>/repeat_<repeat>/`.
The stage is left out so identical conditions can reuse one completed fit.
Each fit has `best_model.bson`, `result.jld2`, `training_history.tsv`, progress
snapshots in `checkpoint.jld2` and interval scores in `diagnostics/metrics.tsv`.
Per-step diagnostic TSVs are kept only for full adaptation with uniform drag,
layout `both`, no noise and objectives `balanced`, `long` or `one_step`.
The rank pilot also saves its final model for calculating update spectra.

## 02: gauge count and curriculum

```powershell
julia --project=. thesis_experiments/02_gauge_density_curriculum.jl
```

This uses full adaptation and uniform drag. `two_gauges` uses cells 2 and 99.
It uses lead 128 if the fold selected `long` in 01 and balanced leads otherwise.
`curriculum_fast` keeps four gauges and adds leads at fractions 0.05, 0.10
and 0.15 of training. Both cases use the shared 200-epoch budget.
Saved controls from 01 provide the four/eight-gauge and original curriculum comparisons.

Outputs go to `results/experiments/gauge_curriculum/`.
`indices/reported_cases.tsv` lists fitted cases and costs.
Each run has `evaluation/interval_metrics.jld2` with key `rows` and a matching TSV.
Top-level `event_metrics.tsv`, `event_means.tsv` and `summary.tsv` contain scores
at each aggregation level. `comparisons.tsv` contains paired event comparisons.

## 03: source control and transfer

```powershell
julia --project=. thesis_experiments/03_source_domain_control.jl
```

This fits full adaptation on `control.jld2`, which has no hidden drag.
It uses endpoint loss at lead 128 with gradients through the full rollout.
The same saved models are then tested on `uniform.jld2` without further training.

Source results go to `results/experiments/source_domain_control/`.
Fits are under `runs/long128/fold_<fold>/repeat_<repeat>/`.
The main tables are `run_index.tsv`, `event_metrics.tsv`, `event_means.tsv`
and `summary.tsv`. Each fit also has interval diagnostics.
Transfer results go to `results/experiments/source_to_target/` as
`event_metrics.tsv` and `summary.csv`. They compare target errors with the
pretrained model on the same events.

## 04: adaptation operation costs

```powershell
julia --project=. thesis_experiments/04_profile_adaptation_cost.jl
```

This loads saved models from 01 and runs no new adaptation fits.
It measures effective-kernel construction, forward loss, forward and reverse
passes, optimizer updates, initial-state regeneration and validation.
It uses one shared batch, three warmups, ten samples, fold 1, repeat 1
and two BLAS threads. `methods`, `samples`, `warmup` and `fold` are at the top.

Outputs go to `results/experiments/adaptation_profiling/`:

- `samples.tsv`: elapsed seconds, allocated bytes and garbage-collection seconds.
- `summary.tsv`: median time, lower/upper quartiles and median allocated bytes.
- `environment.txt`: Julia version, CPU, thread counts and batch details.

These isolated measurements include memory costs and differ from cumulative
training times recorded inside `train_model!`.

## Helper files

`experiment_dependencies.jl` loads `Surge`, diagnostics and common packages.
The experiment-specific helpers are in `thesis_experiments/src/`:

| File | Main job |
|---|---|
| `source_pretraining.jl` | Save source progress and reuse completed fits |
| `sparse_campaign.jl` | Shared experiment paths, fit records, evaluation and summaries |
| `gauge_curriculum_analysis.jl` | Gauge/curriculum summaries and paired event comparisons |
| `source_control_analysis.jl` | Source summaries and transfer evaluation with cached baseline scores |
| `adaptation_profiling.jl` | Separate timed operations, warmups, samples and summaries |

Reusable model, training, forecasting and error calculations remain in the root
`src/` folder. The runners mainly define settings and the order of work.

## Reusing results

Completed fits are reused only when saved settings and splits match.
Existing evaluation tables are also reused. Progress snapshots do not resume
incomplete training. Use a new output directory when changing a dataset, model
or settings, including rank-selection inputs.

The scripts write new results under `results/experiments/` by default.
They do not update `results/reported/`. The Python figure scripts read that saved
results folder, so running a new experiment does not automatically change the figures.
