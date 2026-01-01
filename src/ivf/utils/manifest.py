"""
Run manifest writer for reproducibility.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Dict

import yaml
from omegaconf import OmegaConf

from ivf.config import resolve_config_dict


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _hash_dataset_config(cfg_path: str) -> str:
    try:
        data_cfg = OmegaConf.load(cfg_path)
        root_dir = str(data_cfg.get("root_dir", ""))
        csv_path = str(data_cfg.get("csv_path", ""))
        return _hash_value(f"{root_dir}|{csv_path}")
    except Exception:
        return _hash_value(cfg_path)


def write_run_manifest(cfg, checkpoints: Dict[str, Path], outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / "run_manifest.yaml"

    config_snapshot = resolve_config_dict(cfg)

    dataset_paths = {}
    for key in ["blastocyst_config", "humanembryo2_config", "quality_config", "hungvuong_config"]:
        cfg_path = getattr(cfg.data, key)
        dataset_paths[key] = _hash_dataset_config(cfg_path)

    manifest = {
        "git_commit": _safe_git_commit(),
        "seed": cfg.seed,
        "configs": config_snapshot,
        "dataset_paths_hashed": dataset_paths,
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
