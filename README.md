# Hybrid surge model experiments

This project trains neural models for one-dimensional water levels and velocities.
It adapts pretrained models using sparse water-level observations and compares
forecasts with numerical reference data.

## Project structure

| Folder or file | What it contains |
|---|---|
| `src/` | Models, data preparation, training, evaluation and numerical simulation |
| `thesis_experiments/` | Numbered experiment scripts and experiment-specific helpers |
| `Starting_networks/` | Julia notebooks for training, adaptation, rank selection and plots |
| `Starting_models/` | Numerical simulation notebooks and their separate Julia environment |
| `Data/` | JLD2 datasets and saved data-generation settings |
| `Trained_models/` | Pretrained models saved as BSON files |
| `results/reported/` | Saved results used by the figure scripts |
| `analysis/` | Python scripts that read saved results and make figures |
| `docs/figures/` | Figures and tables made by the Python scripts |
| `Project.toml` and `Manifest.toml` | Julia packages for neural models and experiments |

New experiment results go to `results/experiments/`. Notebook outputs normally go
to `results/notebooks/`. These folders are created when needed.

## Setup

The root manifest was saved with Julia **1.12.6**. The project allows Julia 1.11
and later 1.x versions. Use 1.12.6 to match the manifest.
The main packages are Flux, ComponentArrays, BSON, JLD2, FFTW, Plots and
ProgressMeter. Their versions are recorded in `Manifest.toml`.
From the repository root, install them with:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Numerical data generation uses `Starting_models/Project.toml`. It adds
OrdinaryDiffEq and has its own manifest saved with Julia **1.12.4**.

```powershell
julia --project=Starting_models -e 'using Pkg; Pkg.instantiate()'
```

The `.ipynb` notebooks need a Julia kernel in Jupyter, such as IJulia.
The notebook interface is not included in these project dependencies.
Run cells from top to bottom and check settings before running training
or data-generation cells.

## Source files

Load `src/Surge.jl` to use the neural models. It includes the first seven
implementation files below inside the `Surge` module.

| File in `src/` | Main job |
|---|---|
| `models.jl` | Residual and conservative CNNs, frozen layers, FiLM, LoRA and adaptation settings |
| `io.jl` | Load and save BSON models and text tables |
| `data_utils.jl` | Load datasets, encode states, make one-step pairs, pack sparse observations and split events |
| `training_utils.jl` | Sparse rollout losses, initial-state caches, validation, timing and adaptation training |
| `supervised_training.jl` | One-step source training, input noise and the learning-rate schedule |
| `evaluation_utils.jl` | Forecast events and save spatial errors and example forecasts |
| `experiment_utils.jl` | Prepare training splits, fit adaptations and save or reuse completed runs |
| `experiment_diagnostics.jl` | `ExperimentDiagnostics` module for per-step and interval forecast errors |
| `diagnostic_aggregation.jl` | Average repeats and events inside `ExperimentDiagnostics` |
| `rank_selection.jl` | `LoRARankOptimisation` module for nested LoRA rank selection |
| `plotting_utils.jl` | `ExperimentPlots` module for error maps, forecasts, animations and loss curves |
| `surge_equations.jl` | Physical surge equations and operations on the staggered grid |
| `numerical_dataset.jl` | `NumericalDataset` module for numerical event generation using those equations |

The optional modules are loaded separately. Load `Surge` before diagnostics,
rank selection or plotting because these modules use it. For example, from the
repository root:

```julia
include("src/Surge.jl")
import .Surge
include("src/experiment_diagnostics.jl")

dataset = Surge.load_surge_dataset("uniform.jld2", data_path="Data/thesis_v4")
model = Surge.load_model("Trained_models/flux_correction_pretrained.bson")
```

`Surge.train_model!` has two methods. Sparse event datasets use the method in
`training_utils.jl`. Named tuples with `input` and `target` arrays use the
supervised method in `supervised_training.jl`.

Model input has shape `(N + 1, 5, batch)`. The five channels are padded water
level, face velocity, depth divided by 30, wind stress and boundary flags.
Output has shape `(N + 1, 2, batch)` for water level and velocity.
Water levels use N cells and have a zero padding row. Velocities use N + 1 faces.
Dataset loading keeps initial states in `solution`. Its `times` field gives
elapsed prediction times in seconds and leaves out the initial time.

## Inputs and settings

The numbered adaptation scripts use
`Trained_models/flux_correction_pretrained.bson` and datasets in `Data/thesis_v4/`.
Source pretraining uses `Data/24_hour_source.jld2`.
Some notebooks instead use `Data/drag_target_10.jld2` or
`Trained_models/drag_source_pretrained.bson`. Check each notebook's settings cell.

Settings are near the top of each runner or in notebook settings cells.
Check input paths, output paths, model choice, event splits, random seeds,
station indices, observation noise, rollout leads, epochs and optimizer settings.
Runner paths are based on the script location and do not depend on the current
working directory. The commands here assume the repository root.

To run the main experiments:

```powershell
julia --project=. thesis_experiments/01_sparse_adaptation_experiments.jl
```

This runs many training fits. See the
[experiment README](thesis_experiments/README.md) for all five commands,
required inputs, experiment order and output files. Each runner starts when
executed directly. Including it only loads definitions. Use separate Julia
sessions for different runners because some constants have the same names.

The dataset-generation notebook is
`Starting_models/generate_experiment_3_datasets.ipynb`. Its current output is
`Data/experiment_3/<profile>.jld2`, while the numbered runners read
`Data/thesis_v4/<profile>.jld2`. Check `drag_profile`, `output_file` and generation
settings before making new data. Saved settings for existing experiment datasets
are in `Data/thesis_v4/generation_configuration.toml`.

## Saved outputs

Adaptation runs save the best model, training history, run settings and progress
snapshots. Completed fits are reused when their settings and splits match.
Incomplete fits restart from the beginning. Progress snapshots are not resumed.
Use a new output directory when changing settings. Keep existing input data,
pretrained models and `results/reported/` files when comparing with saved results.

Python figure scripts need NumPy and Matplotlib. To regenerate figures from
`results/reported/`:

```powershell
python analysis/reproduce_figures.py
```

This writes to `docs/figures/` and can replace existing figure files.
Shared Python readers and plotting helpers are in `analysis/results_common.py`.
