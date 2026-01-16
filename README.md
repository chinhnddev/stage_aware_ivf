# IVF Embryo Quality - Stage-aware Morphology-informed Learning

## Label schema & guardrails
- **Tasks**: (T1) Blastocyst morphology (EXP/ICM/TE multi-head classification), (T2) Stage classification (cleavage/morula/blastocyst), (T3) Quality prediction (good/poor).
- **Quality mapping**: good -> `4AA`, `5AA`, or any `3-4AB` (e.g., `3AB`, `4AB`); poor -> any `<=3CC` (e.g., `1CC`, `2CC`, `3CC`); all others -> `unknown` (excluded from quality training unless explicitly included).
- **Morphology targets**: expansion classes 1–6, ICM grades A/B/C, TE grades A/B/C; IDs are zero-based in code and generated via `ivf/data/label_schema.py`.
- **Guardrails (hard rules)**: day metadata (`day`, `day_label`, `day_str`) must never be present in model inputs, and no segmentation/mask inputs are allowed. Violations raise errors via `ivf/utils/guardrails.py`.
- **Source of truth**: label definitions, parsing, and mappings live in `src/ivf/data/label_schema.py`; guardrail checks live in `src/ivf/utils/guardrails.py`.
- **External test policy**: Hung Vuong hospital dataset is strictly external-test only; adapters and split code refuse any train/val usage.

## Architecture
- Shared encoder (`ConvNeXtMini`) -> pooled feature vector.
- Heads: Morphology (EXP/ICM/TE), Stage (cleavage/morula/blastocyst), Quality (good/poor).
- Morphology-conditioned quality: quality head receives features + morphology + stage probabilities (concat by default; FiLM optional) so quality is informed by predicted morphology and stage. No day metadata is ever accepted by the model forward.
- Outputs: `{"features": f, "morph": {"exp":..., "icm":..., "te":...}, "stage":..., "quality":...}` with losses defined in `src/ivf/losses.py`.
- Encoder initializes randomly by default; optional `weights_path` can load external weights, but no ImageNet-pretrained claim is made.

## EmbryoNetLite (from scratch)
- Lightweight MBConv+SE encoder trained from scratch (no pretrained weights).
- Switch via config: set `model.encoder_config: configs/model/encoder_embryonet_lite.yaml`.
- Phase reminders: morphology heads train only in Phase 1 (Day-5). Day-3 never uses morphology heads.

Example YAML snippets per phase:
```yaml
# Phase 1 (morphology)
model:
  encoder_config: configs/model/encoder_embryonet_lite.yaml
  heads:
    quality_conditioning: none
```
```yaml
# Phase 2 (stage)
model:
  encoder_config: configs/model/encoder_embryonet_lite.yaml
```
```yaml
# Phase 3 (joint)
model:
  encoder_config: configs/model/encoder_embryonet_lite.yaml
```
```yaml
# Phase 4 (quality)
model:
  encoder_config: configs/model/encoder_embryonet_lite.yaml
```
Note: Phase-4q (latent q-score) is archived by default because morphology GT is too sparse. Enable with `model.use_q_score: true` if needed.

## Training order & why
- Phase 1 (morphology): learn EXP/ICM/TE on blastocysts with light augmentations; establishes morphology features.
- Phase 1 checkpointing uses `val/te_macro_f1` (fallback `val/te_bal_acc`) to reduce class-collapse risk.
- Phase 2 (stage): learn stage with medium augmentations; encoder progressively unfrozen for stage awareness.
- Phase 3 (joint): light joint adaptation on blastocyst + HumanEmbryo2.0; balanced morphology/stage losses.
- Phase 4 (quality): train only the morphology+stage-conditioned quality head; encoder, morphology, and stage heads frozen.
- Phase 4q (q-score) is disabled by default; enable only for research experiments with sufficient morphology GT.
- Augmentations are applied ONLY in train loaders; val/test/external always use eval transforms.
- Pipeline overview: see `docs/PIPELINE.md`.

## Main Experiment
- Train sources: blastocyst + HumanEmbryo2.0 through phases 1–4; final evaluation uses `phase4_quality.ckpt`.
- External test policy: Hung Vuong hospital data is external-only and never used in training.
- Metrics on external: AUROC, AUPRC, F1; reported overall plus day-3 and day-5 slices.
- External evaluation modes: zero-shot (source threshold from EXP-4 val) and calibrated (threshold/temperature tuned on HV-val only).
- Day is used only for analysis/reporting; it is never passed to the model.
- Run end-to-end: `python scripts/run_main_experiment.py`

## Main Experiment Summary
- Architecture proposed in this repo is trained from scratch by default; optional external weights can be loaded without changing claims.
- No day leakage: day metadata is only used for reporting slices and never passed to the model.
- No hospital data in training: Hung Vuong is strictly external evaluation only.

## Reproducibility Checklist
- `set_global_seed()` called in all entrypoints; deterministic mode enabled by default.
- Structured configs validated via OmegaConf (unknown keys raise).
- All outputs routed to `outputs/` with checkpoints, logs, and reports separated.
- Run manifest saved to `outputs/run_manifest.yaml` with config snapshot, seed, dataset path hashes, and checkpoints.
- Train-only augmentations; val/test/external strictly use eval transforms.

## Developer Commands
- Install dev tooling: `make dev` (or `pip install -e ".[dev]"`)
- Tests: `make test`
- Format: `make fmt`
- Lint: `make lint`
- Full check: `make check`

## Synthetic Smoke Test
- Generate a tiny synthetic dataset and run all phases + external eval:
  `python scripts/smoke_e2e.py --config configs/experiment/smoke.yaml --fast`
- Output artifacts are written under `outputs/` and validated by the script.

## Dataset Metadata Status
- Blastocyst (Kaggle): metadata CSV `data/metadata/blastocyst.csv`; columns `image_path`, `grade`, `gardner`, `exp`, `icm`, `te`, `stage`, `quality`, `day`, `embryo_id`, `patient_id`, `split`, `dataset`; missing fields: `day`, `patient_id`, `stage`, `quality`; readiness for training: YES (splits fall back to image_id when `patient_id` is missing).
- HumanEmbryo2.0: metadata CSV `data/metadata/humanembryo2.csv`; columns `image_path`, `stage`, `day`, `grade`, `gardner`, `quality`, `embryo_id`, `patient_id`, `split`, `dataset`; missing fields: `day`, `patient_id`, `grade`, `gardner`, `quality`; readiness for training: YES.
- Quality public: metadata CSV `data/metadata/quality_public.csv`; columns `image_path`, `grade`, `gardner`, `quality`, `exp`, `icm`, `te`, `day`, `embryo_id`, `patient_id`, `split`, `dataset`; missing fields: `day`, `patient_id`; readiness for training: YES (samples with unknown quality mapping are filtered).
- Hung Vuong: metadata CSV `data/metadata/hungvuong.csv`; columns `image_path`, `quality`, `quality_raw`, `day`, `stage`, `grade`, `gardner`, `embryo_id`, `patient_id`, `split`, `dataset`; missing fields: `stage`, `grade`, `gardner`, `patient_id`; readiness for training: YES (external test only).
