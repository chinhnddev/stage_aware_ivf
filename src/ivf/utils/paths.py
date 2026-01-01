"""
Path helpers for enforcing output structure.
"""

from pathlib import Path


def ensure_outputs_dir(path: str) -> Path:
    resolved = Path(path)
    if "outputs" not in resolved.parts and not str(path).startswith("outputs"):
        raise ValueError(f"Outputs must be written under outputs/: {path}")
    return resolved
