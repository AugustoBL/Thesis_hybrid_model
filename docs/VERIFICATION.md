# Local verification — 20 October 2026

All checks ran against the standalone copy. Nothing was pushed or created on GitHub, and no local Git repository was initialized in the copy.

## Passed checks

- Julia 1.12.6 loaded the existing locked Flux/JLD2/BSON environment. The manifest records Julia 1.12.4; that exact Julia patch release was not used for these migration checks. The manifest has no local-path dependency entries.
- All Julia files parsed, and all five experiment runners launched in isolated plan-only modules: **23 assertions passed**.
- Both restored shared helpers and rank-selection module imported successfully. All four 18-event ensembles and the 40-event source-pretraining ensemble loaded; time coverage and grid dimensions passed checks. The 2,706-parameter pretrained model produced finite short forecasts; all six adaptation methods produced finite losses and gradients: **37 assertions passed**.
- All five fold method/temporal selection files and selected ranks were present. All 360 indexed main fit metadata/model pairs loaded, with saved configuration/split records and positive trainable-parameter counts: **1,451 assertions passed**.
- Python checked **19 local Julia include expressions**, parsed all retained Python files, checked all 390 main run-index references, and verified that each required dataset is actual data rather than an undownloaded LFS pointer.
- Saved equal-event aggregation reproduced **208 main**, **48 selected follow-up**, **8 source-control**, and **8 target-evaluation** summary groups. The source-fine-tuned target errors reproduced the thesis's **99.55/127.81 mm** values.
- All seven numerical figures referenced by the active results chapter/appendix regenerated from saved data. Their numerical aggregates were checked; this does not claim pixel-identical thesis typography. Generated figures are ignored by Git.
- SHA-256 hashes confirmed that all five required input datasets and the selected pretrained checkpoint are byte-identical to the source working tree.
- Pattern scans over the selected text and binary files detected **zero** GitHub-token, AWS-key, private-key, embedded URL-credential or assigned-secret patterns. No `.env`, private-key or credential files were selected. This is a targeted local scan, not a guarantee against every possible secret format.
- No ordinary-Git file exceeds 100 MiB. All `.jld2`/`.bson` files are configured for LFS, including the four required ensembles larger than that limit.
- Source branch and full porcelain working-tree status matched the pre-preparation snapshot. Original repository files, Git metadata and branches were not modified by preparation. The copy has neither a `.git` entry nor a `thesis/` directory.

## Re-run

```powershell
python scripts/verify_repository.py
python analysis/reproduce_figures.py
julia --project=Starting_networks scripts/verify_runtime.jl
```

The Python verifier also checks the supplied file checksum manifest. Changing a prepared file intentionally requires regenerating the corresponding checksum or removing the snapshot manifest before treating the project as a new snapshot. Fresh outputs under `results/thesis_reproduction/` and regenerated figures/caches are outside the snapshot.

The complete experiments and their long forecast evaluations were not rerun. Fresh-environment installation and remote Git LFS upload/download were not tested: packages were already installed locally, GitHub CLI was absent from PATH, and publication was explicitly deferred to the user.
