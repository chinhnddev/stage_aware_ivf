"""
Compare image intensity distributions between training and external datasets.

Usage:
    python scripts/compare_domain_shift.py \
        --train-config configs/data/quality_public.yaml \
        --external-config configs/data/hungvuong.yaml \
        --output outputs/reports/domain_shift_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image


def _normalize_path(path_value: str, root_dir: str) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    root = Path(str(root_dir).replace("\\", "/"))
    if not path.is_absolute():
        path = root / path
    return path


def _summarize_dataset(csv_path: str, root_dir: str, image_col: str, max_samples: int, seed: int, name: str):
    df = pd.read_csv(csv_path)
    if image_col not in df.columns:
        raise ValueError(f"Missing image_col '{image_col}' in {csv_path}")
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    sum_rgb = np.zeros(3, dtype=np.float64)
    sumsq_rgb = np.zeros(3, dtype=np.float64)
    sum_gray = 0.0
    sumsq_gray = 0.0
    pixel_count = 0
    hist_bins = 32
    hist = np.zeros(hist_bins, dtype=np.int64)
    missing = 0

    for _, row in df.iterrows():
        path = _normalize_path(row[image_col], root_dir)
        if not path.exists():
            missing += 1
            continue
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue
        h, w, _ = arr.shape
        count = h * w
        pixel_count += count
        sum_rgb += arr.reshape(-1, 3).sum(axis=0)
        sumsq_rgb += (arr.reshape(-1, 3) ** 2).sum(axis=0)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        sum_gray += float(gray.sum())
        sumsq_gray += float((gray ** 2).sum())
        hist += np.histogram(gray, bins=hist_bins, range=(0, 255))[0]

    if pixel_count == 0:
        raise ValueError(f"No valid images found for {name}.")

    mean_rgb = (sum_rgb / pixel_count).tolist()
    std_rgb = (np.sqrt(np.maximum(sumsq_rgb / pixel_count - np.square(mean_rgb), 0.0))).tolist()
    mean_gray = sum_gray / pixel_count
    std_gray = float(np.sqrt(max(sumsq_gray / pixel_count - mean_gray ** 2, 0.0)))
    hist_frac = (hist / hist.sum()).tolist()

    return {
        "name": name,
        "num_images": int(len(df)),
        "missing_images": int(missing),
        "pixel_count": int(pixel_count),
        "mean_rgb": [float(x) for x in mean_rgb],
        "std_rgb": [float(x) for x in std_rgb],
        "mean_gray": float(mean_gray),
        "std_gray": std_gray,
        "hist_bins": hist_bins,
        "hist_counts": hist.tolist(),
        "hist_frac": [float(x) for x in hist_frac],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare domain shift stats.")
    parser.add_argument("--train-config", default="configs/data/quality_public.yaml")
    parser.add_argument("--external-config", default="configs/data/hungvuong.yaml")
    parser.add_argument("--output", default="outputs/reports/domain_shift_stats.json")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("domain_shift")

    train_cfg = OmegaConf.load(args.train_config)
    ext_cfg = OmegaConf.load(args.external_config)

    train_stats = _summarize_dataset(
        csv_path=train_cfg["csv_path"],
        root_dir=train_cfg["root_dir"],
        image_col=train_cfg["image_col"],
        max_samples=args.max_samples,
        seed=args.seed,
        name="train",
    )
    ext_stats = _summarize_dataset(
        csv_path=ext_cfg["csv_path"],
        root_dir=ext_cfg["root_dir"],
        image_col=ext_cfg["image_col"],
        max_samples=args.max_samples,
        seed=args.seed,
        name="external",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": train_stats, "external": ext_stats}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved domain shift stats to %s", output_path)


if __name__ == "__main__":
    main()
