# Polymer formulation candidate libraries

This V47 release candidate contains research code and frozen protocol settings for Supporting Information S79–S82. The MIT license applies to the included author code. Third-party data, trained weights, generated libraries, and trajectory archives are excluded.

## Setup and startup checks

Use Python 3.10 and install the dependencies with python -m pip install -r requirements.txt. The startup checks used Python 3.10.20, NumPy 2.2.6 and PyTorch 2.11.0+cu128 in a fresh virtual environment that inherited the existing dependency installation.

- S79: python S79/scripts/run_mpg_ga.py --help
- S80: python S80/scripts/run_case.py --help
- S81: python S81_S82/scripts/run_traced_ga.py --help
- S82: python S81_S82/scripts/run_marginal_contrasts.py --help

The four corresponding analysis scripts also passed import checks. These checks do not reproduce the paper's numerical results.

## Frozen inputs for numerical replay

All paths below are relative to the repository root. Use the archived files from the exact experiment versions named below, preserving filenames and replicate numbers (1–5). Source DOIs identify reference data; they do not supply the processed evaluator tables, archived seeds, or model weights. Consult the input audit and source mapping in SI S79–S82 before running a replay.

| Workflow | Origin | Place inputs here |
| --- | --- | --- |
| S79 industrial PP | V35 experiment archive, SI S79 | Copy its inputs tree into `S79/inputs/`, including `industrial_pp_100_recipes.csv` and `best_feasible_per_successful_start.npz`. |
| S80 supplementary cases | V38 two-case experiment archive, SI S80 | Copy its inputs tree into `S80/inputs/`, including `data/pp_hdpe_banana_recipes.csv` and `data/pcm_material_designs.csv`. |
| S81 traced GA | V42 trajectory replay archive, SI S81 | Copy its inputs tree into `S81_S82/inputs/`. Preserve `industrial_pp_100_recipes.csv`, `best_feasible_per_successful_start.npz`, and `historical_formal/replicate_N/` (upstream candidates and archived seed indices, including `MPG_selected.npy`). |
| S82 marginal controls | V42 marginal-control archive, SI S82 | Use the same V42 inputs tree, including `mpga_starts/replicate_N.npz` and `generator_models/replicate_N.pt`. |

The workflows also expect these shared archives:

- **Historical GA and random starts** — `PES_constraint_synthesis_no_anchor_20261010_v1/results_raw/formal_v1/CASE/replicate_N/`. Preserve `methods/G_GA/library.npz`, `methods/R_GA/library.npz`, and the corresponding accepted-candidate records for the S81 replay. S80 reads `upstream/R/candidates.npz` for both supplementary cases. Use case keys `industrial_pp_100`, `natural_fiber_pp_hdpe` and `pcm_polymer_composite`.
- **V35 MPG–GA comparator for S81** — `CMS_v35_three_seed_ga_routes_20261104_v1/results/formal/replicate_N/MPG_GA/`. Keep the original results tree, including the archived accepted candidates.
- **V16 MPGA trajectories for S82** — `CMS_v16_focused_analysis_20261014_v1/open_source_package/results_raw/mpga_t_paired_v16_formal/formal_v1/replicate_N/`. Keep `same_rerun_best_endpoint_candidates.npz` and `trajectory_feasible_candidates.npz`.
- **Frozen bootstrap indices** — `PES_v11_review_supplement_20261012_v1/results_summary/case_modules/CASE/B_bootstrap_indices_CASE.npz` for S80 and S82. The S79 historical evaluator helper locates its industrial bootstrap file under `output_cms_v29/reproducibility/three_route/full_archive/*/results_summary/case_modules/industrial_pp_100/`.
- **Analysis-only replay** also requires the corresponding archived outputs under each workflow's `results/` tree. For S82 retain the V42 formal and cost-supplement outputs as well as the GA replay proposals.

Reference data provenance: industrial PP is from the supporting workbook of Joo et al. (10.1002/int.22700); natural-fiber inputs are from Tables A1–A2 of Espinach et al. (10.1016/j.jclepro.2024.144244); the PCM reference dataset is version 1 (10.17632/p63pgr2hb9.1). These sources are separate from this code repository.

## Verification status

The GitHub download matched the local snapshot byte for byte before this README update. Startup and import checks passed with inherited dependencies. A fresh isolated environment was created, but its dependency installation could not obtain NumPy 2.2.6 from the package index available in the test environment. Full replay of the paper's results has not been verified. This candidate has no public release or DOI.