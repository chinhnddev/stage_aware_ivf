# Project Structure (stage_aware_ivf)

This map documents the core layout, reproducible entry points, and output conventions.

## Top-Level Directories

- `src/ivf/`: Core library (models, data, losses, metrics, training utilities).
- `scripts/`: Entry points for training, evaluation, and data prep.
  - Core: `train_phase.py`, `eval_external.py`
  - Data prep: `prepare_*.py`, `make_quality_splits.py`, `merge_blastocyst_gardner.py`
- `configs/`: Experiment and model configuration.
  - `configs/experiment/`: Train/eval configs used by CLI
  - `configs/model/`: Encoder and head definitions
  - `configs/data/`: Dataset definitions
- `data/`: Raw datasets + prepared metadata/splits.
  - `data/metadata/`: CSV metadata for each dataset
  - `data/processed/splits/`: Split CSVs (train/val/test)
- `outputs/`: Checkpoints, reports, and logs (ignored by git).
- `docs/`: Research notes and pipeline references.
- `tests/`: Unit tests and sanity checks.

## Phase Map (Reproducible Entry Points)

- Phase 1 (morphology):
  - `python scripts/train_phase.py --phase morph --config configs/experiment/exp1_morph.yaml`
- Phase 2 (stage):
  - `python scripts/train_phase.py --phase stage --config configs/experiment/embryonet_lite.yaml`
- Phase 3 (quality):
  - `python scripts/train_phase.py --phase quality --config configs/experiment/phase3_quality_embryonet_lite.yaml`
- Phase 4 (q-score):
  - `python scripts/train_phase.py --phase q --config configs/experiment/phase4_q_embryonet_lite.yaml`
  - In-domain eval:
    - `python scripts/eval_external.py --eval_in_domain_q --config configs/experiment/phase4_q_embryonet_lite.yaml`
- Phase 5 (cross-domain):
  - `python scripts/eval_external.py --config configs/experiment/phase3_quality_embryonet_lite.yaml`
  - `python scripts/eval_external.py --q_only --config configs/experiment/phase4_q_embryonet_lite.yaml`

## Outputs (Paper-Grade vs Temporary)

Paper-grade outputs:
- `outputs/checkpoints/phase*/*`
- `outputs/reports/in_domain/*`
- `outputs/reports/cross_domain/*`

Temporary / debug:
- `outputs/logs/`
- `outputs/tmp/`
- `outputs/debug/`

## Cleanup / Hygiene

Use `scripts/cleanup_project.py` to generate a dry-run cleanup plan and optionally apply
archive moves. The script requires `CLEANUP_CONFIRM=true` to apply any changes.
