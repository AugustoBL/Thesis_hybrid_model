# Sparse-observation surge forecasting

Julia experiments for adapting a pretrained shallow-water forecasting model using sparse water-level observations. This standalone project is a selected snapshot of the `thesis_hybrid_model` working tree, including its restored experiment helpers. It includes the code, original inputs, selected model weights and saved evidence used by the active thesis results chapter and reproducibility appendix. The thesis document itself is excluded.

## Structure

| Directory | Purpose |
|---|---|
| `thesis_experiments/` | Five commented runners and common result-saving code |
| `thesis_experiments/shared/` | Dataset/pretraining helper and follow-up fitting/evaluation runtime |
| `src/` | Model architectures, sparse observations, training, forecasting and diagnostics |
| `lora_rank_optimisation/` | Nested inner-validation LoRA rank selection |
| `Starting_networks/` | Pinned Julia package environment; historical notebooks are excluded |
| `Data/thesis_v4/` | Four original paired source/target ensembles and generation configuration |
| `Data/24_hour_source.jld2` | Original source-pretraining ensemble |
| `Trained_models/` | Original selected flux-correction pretrained checkpoint |
| `results/thesis/` | Selected immutable historical evidence; see [results provenance](docs/RESULTS_PROVENANCE.md) |
| `analysis/` | Saved-results figure builders and their Python requirements |
| `scripts/` | Dependency, input and saved-result verification |
| `docs/` | Provenance, migration details and verification records |

## Setup

Install Git LFS before cloning or pulling the large data files. After cloning:

```powershell
git lfs install --local
git lfs pull
julia --project=Starting_networks -e 'using Pkg; Pkg.instantiate()'
```

Julia **1.12.4** matches the saved manifest and historical profiling environment. Migration checks were run with Julia **1.12.6**, using the existing locked package versions. The environment includes Flux, BSON, JLD2, ComponentArrays, Plots and ProgressMeter. Keep `Project.toml` and `Manifest.toml` together. Use a working Julia executable on PATH.

All five required datasets are included through Git LFS, including the four 110,130,085-byte target/control ensembles. No substitute data or sibling repository is required. LFS pointer files are not usable datasets: run `git lfs pull` to obtain the actual files. The original numerical solver, generation notebooks and per-event generation cache are excluded because the retained experiments consume these prepared ensembles. Dataset settings are recorded in `Data/thesis_v4/generation_configuration.toml`; the original ensemble files do not record solver tolerances.

## Run experiments

Run from the repository root. Omit `--run` to print a plan without loading data or training:

```powershell
julia --project=Starting_networks thesis_experiments/01_main_experiments.jl
```

To execute:

```powershell
julia --project=Starting_networks thesis_experiments/01_main_experiments.jl --run
julia --project=Starting_networks thesis_experiments/02_gauge_and_curriculum_followups.jl --run
julia --project=Starting_networks thesis_experiments/03_no_domain_shift_control.jl --run
julia --threads=1 --project=Starting_networks thesis_experiments/04_profile_adaptation.jl --run
```

Run 01 before 02 and 04; they read its saved selections/models. Runner 03 is independent. Profiling uses one Julia thread and sets two BLAS threads internally. To profile or run follow-ups against the included historical main fits instead, change `baseline_directory` near the top of runners 02/04 to `joinpath(repository, "results", "thesis", "thesis_flux_correction_v4")`. The required selections, main manifest, fit metadata and selected weights are included. Their default remains the newly reproduced main campaign.

Optional source pretraining:

```powershell
julia --project=Starting_networks thesis_experiments/00_source_pretraining.jl --run
```

New runs write to ignored `results/thesis_reproduction/`. They never overwrite historical results by default. Completed fits are reused and interrupted fits resume their checkpoints. Change the output directory when changing scientific settings. New pretraining saves a separate checkpoint; change `pretrained_file` explicitly to use it. Keep the original checkpoint for comparisons with the reported thesis results.

These commands perform substantial computation. The code reflects the source working tree at migration: historical numerical trajectories and hardware timings are not promised to be identical. In particular, the `one_step` output label represents the current endpoint-128, 127-detached-step implementation, rather than the historical cached-predecessor mechanism. See the [runner guide](thesis_experiments/README.md) for scientific settings and saved-file formats.

## Reproduce saved figures and check files

Python 3.10 was used for the migration checks. Figure generation only reads saved error tables; it does not train or run forecasts.

```powershell
python -m pip install -r analysis/requirements.txt
python scripts/verify_repository.py
python analysis/reproduce_figures.py
julia --project=Starting_networks scripts/verify_runtime.jl
```

The seven active numerical result figures are generated in ignored `docs/figures/`, including parameter accuracy, temporal accuracy/cost, observation geometry, noise sensitivity, observed/interior error histories, and paired trajectory differences. No thesis source is needed. The Python verifier recomputes saved equal-event aggregates and checks local file dependencies; the Julia verifier parses all Julia code, launches all five plan modes, imports restored helpers, loads the five datasets and model checkpoints, and exercises short forecasts and gradients. Neither verifier launches the full training campaigns.

Text run indices use repository-relative paths. Historical binary metadata retains its original scientific records, including training-machine paths; runners and plot builders resolve current inputs and outputs from the repository location. Do not replace historical manifests with hashes from a new campaign.

## Publication

See [PUBLISH.md](PUBLISH.md) for reviewable PowerShell commands to initialize a fresh history and create a public GitHub repository. No repository creation or publication was performed during preparation. Git LFS is required for the included binary inputs. No license has been added: this snapshot does not invent licensing terms for the original work.
