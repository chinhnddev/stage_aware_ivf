import subprocess
import sys
from pathlib import Path


def test_smoke_e2e_fast():
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "smoke_e2e.py"),
        "--config",
        "configs/experiment/smoke.yaml",
        "--fast",
        "--device",
        "cpu",
        "--num_workers",
        "0",
        "--seed",
        "123",
    ]
    subprocess.run(cmd, check=True)
