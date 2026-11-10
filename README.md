# Polymer formulation candidate libraries

This V47 release candidate contains research code and frozen protocol settings for Supporting Information S79–S82. The MIT license applies to the included author code. Third-party data, trained weights, generated libraries, and trajectory archives are excluded.

## Setup and startup checks

Use Python 3.10 and install the dependencies with python -m pip install -r requirements.txt. The startup checks used Python 3.10.20, NumPy 2.2.6 and PyTorch 2.11.0+cu128 in a fresh virtual environment that inherited the existing dependency installation.

- S79: python S79/scripts/run_mpg_ga.py --help
- S80: python S80/scripts/run_case.py --help
- S81: python S81_S82/scripts/run_traced_ga.py --help
- S82: python S81_S82/scripts/run_marginal_contrasts.py --help

The four corresponding analysis scripts also passed import checks. These checks do not reproduce the paper's numerical results.

## Full replay inputs

Use the input provenance and archive mappings in SI S79–S82. Each workflow reads its frozen protocols from its own configs folder and experiment inputs from its inputs folder. Additional historical archives are expected in the layout named by the scripts, including the earlier generation archive, V16 MPGA results and V35 route outputs. Use the original frozen inputs for numerical comparison. Reference sources are described in the manuscript under Data and code availability.

## Release status

Local candidate. Full replay from a clean dependency installation has not been verified. No public repository or DOI is assigned.
