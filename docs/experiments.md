# Experiments

## ABL-BIN — In-domain Binary Classifier (Hung Vuong)

Purpose: train a supervised in-domain binary classifier on Hung Vuong (HV) only. This is independent from the q-head/morphology/stage pipelines.

### Reproduce
Train (splits HV/train into train/val; HV/test is held out):
```
python scripts/train_abl_bin.py --config configs/experiment/base.yaml --backbone convnext_mini --hv_root <HV_ROOT>
```

Evaluate from checkpoint:
```
python scripts/eval_abl_bin.py --config configs/experiment/base.yaml --ckpt outputs/checkpoints/abl_bin_convnext_mini_seed42.ckpt --hv_root <HV_ROOT> --seed 42
```

### Outputs
- Checkpoint: `outputs/checkpoints/abl_bin_{backbone}_seed{seed}.ckpt`
- Reports: `outputs/abl_bin/`
  - `metrics_test.json`
  - `predictions_test.csv`
  - `best_threshold.json`

### Smoke test
Run a quick sanity check on a small setup:
```
python scripts/train_abl_bin.py --config configs/experiment/base.yaml --backbone convnext_mini --hv_root <HV_ROOT> --max_epochs 1 --batch_size 4
```

Expected logs include:
- `HV split sizes: train=... val=... test=...`
- `HV train labels: total=... kept=... good=... not_good=...`
- `Saved ABL-BIN reports to outputs/abl_bin`

### Notes
- HV/test is never used for training or thresholding. HV/train is split deterministically into train/val and saved to `outputs/splits/hv_trainval_seed{seed}.csv`.
- Threshold is tuned on HV-val only and applied once to HV-test.
- Metrics are reported overall plus day3/day5 if day labels exist.
