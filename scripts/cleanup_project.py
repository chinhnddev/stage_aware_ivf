"""
Project cleanup utility (dry-run by default).

This script classifies files into CORE / ARCHIVE / TRASH, detects duplicate configs,
and proposes a safe re-organization of outputs/ without deleting checkpoints.

Apply mode requires CLEANUP_CONFIRM=true to avoid accidental data loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]

TRASH_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "lightning_logs",
    "wandb",
}
TRASH_FILES = {".DS_Store", "Thumbs.db"}
TRASH_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log", ".tmp"}

CORE_SCRIPT_PREFIXES = ("train_", "eval_", "prepare_", "make_", "merge_")
CORE_SCRIPT_NAMES = {"run_main_experiment.py"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project cleanup (dry-run by default).")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without changing files (default).")
    parser.add_argument("--apply", action="store_true", help="Apply moves/deletions (requires CLEANUP_CONFIRM=true).")
    parser.add_argument("--write-plan", default=None, help="Optional path to write JSON plan.")
    parser.add_argument("--delete-trash", action="store_true", help="Delete trash instead of archiving.")
    parser.add_argument("--max-list", type=int, default=25, help="Max items to print per category.")
    return parser.parse_args()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            yield path


def _is_trash(path: Path) -> bool:
    if any(part in TRASH_DIRS for part in path.parts):
        return True
    if path.name in TRASH_FILES:
        return True
    if path.suffix.lower() in TRASH_SUFFIXES:
        return True
    return False


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _referenced_configs(paths: Iterable[Path]) -> List[str]:
    pattern = re.compile(r"configs/[^\s'\"\\]+\.ya?ml")
    refs = set()
    for path in paths:
        if not path.suffix.lower() == ".py":
            continue
        text = _read_text(path)
        for match in pattern.findall(text):
            refs.add(match.replace("\\", "/"))
    doc_paths = [ROOT / "README.md", ROOT / "docs"]
    for doc_path in doc_paths:
        if doc_path.is_dir():
            for path in doc_path.rglob("*.md"):
                text = _read_text(path)
                for match in pattern.findall(text):
                    refs.add(match.replace("\\", "/"))
        elif doc_path.is_file() and doc_path.suffix.lower() == ".md":
            text = _read_text(doc_path)
            for match in pattern.findall(text):
                refs.add(match.replace("\\", "/"))
    return sorted(refs)


def _hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _find_duplicate_configs(config_paths: List[Path]) -> Dict[str, List[str]]:
    hashes: Dict[str, List[str]] = {}
    for path in config_paths:
        digest = _hash_file(path)
        hashes.setdefault(digest, []).append(str(path.as_posix()))
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}


def _classify_script(path: Path) -> str:
    name = path.name
    if name.startswith(CORE_SCRIPT_PREFIXES) or name in CORE_SCRIPT_NAMES:
        return "core"
    return "archive"


def _outputs_checkpoint_dest(name: str) -> Path:
    lowered = name.lower()
    for phase in ("phase1", "phase2", "phase3", "phase4", "phase5"):
        if phase in lowered:
            return Path("outputs") / "checkpoints" / phase
    return Path("outputs") / "checkpoints" / "misc"


def _outputs_report_dest(name: str) -> Path:
    lowered = name.lower()
    if "in_domain" in lowered:
        return Path("outputs") / "reports" / "in_domain"
    if "exp5" in lowered or "external" in lowered or "hungvuong" in lowered:
        return Path("outputs") / "reports" / "cross_domain"
    return Path("outputs") / "reports" / "misc"


def build_plan() -> Dict[str, List[str]]:
    files = list(_iter_files(ROOT))
    referenced = set(_referenced_configs([p for p in files if p.parent.name == "scripts"]))
    referenced = set(ref.replace("\\", "/") for ref in referenced)

    core: List[str] = []
    archive: List[str] = []
    trash: List[str] = []
    output_moves: List[Tuple[str, str]] = []

    for path in files:
        rel = path.relative_to(ROOT)
        rel_posix = rel.as_posix()

        if _is_trash(path):
            trash.append(rel_posix)
            continue

        if rel.parts[0] == "outputs":
            if rel.parts[1] == "checkpoints":
                core.append(rel_posix)
                if len(rel.parts) == 3:
                    output_moves.append((rel_posix, str(_outputs_checkpoint_dest(path.name) / path.name)))
            elif rel.parts[1] == "reports":
                archive.append(rel_posix)
                if len(rel.parts) == 3:
                    output_moves.append((rel_posix, str(_outputs_report_dest(path.name) / path.name)))
            elif rel.parts[1] == "logs":
                trash.append(rel_posix)
            else:
                archive.append(rel_posix)
            continue

        if rel.parts[0] == "src":
            core.append(rel_posix)
            continue
        if rel.parts[0] == "data":
            core.append(rel_posix)
            continue
        if rel.parts[0] == "docs" or rel_posix in {"README.md"}:
            core.append(rel_posix)
            continue

        if rel.parts[0] == "scripts":
            bucket = _classify_script(path)
            (core if bucket == "core" else archive).append(rel_posix)
            continue

        if rel.parts[0] == "configs":
            if rel_posix.replace("\\", "/") in referenced:
                core.append(rel_posix)
            else:
                archive.append(rel_posix)
            continue

        if path.suffix.lower() in {".ipynb"}:
            archive.append(rel_posix)
            continue

        core.append(rel_posix)

    config_paths = [p for p in files if p.suffix in {".yml", ".yaml"} and "configs" in p.parts]
    duplicates = _find_duplicate_configs(config_paths)

    return {
        "core": sorted(core),
        "archive": sorted(archive),
        "trash": sorted(trash),
        "output_moves": [f"{src} -> {dst}" for src, dst in output_moves],
        "duplicate_configs": json.loads(json.dumps(duplicates)),
        "referenced_configs": sorted(referenced),
    }


def _print_plan(plan: Dict[str, List[str]], max_list: int) -> None:
    def _print_list(label: str, items: List[str]) -> None:
        print(f"{label}: {len(items)}")
        for item in items[:max_list]:
            print(f"  - {item}")
        if len(items) > max_list:
            print(f"  ... {len(items) - max_list} more")

    _print_list("CORE", plan["core"])
    _print_list("ARCHIVE", plan["archive"])
    _print_list("TRASH", plan["trash"])
    _print_list("OUTPUT_MOVES", plan["output_moves"])
    if plan["duplicate_configs"]:
        print("DUPLICATE_CONFIGS:")
        for digest, paths in plan["duplicate_configs"].items():
            print(f"  - {digest[:8]}: {paths}")


def _ensure_archive_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _move_path(src: Path, dst: Path) -> None:
    _ensure_archive_dir(dst)
    if dst.exists():
        raise FileExistsError(f"Destination exists: {dst}")
    shutil.move(str(src), str(dst))


def apply_plan(plan: Dict[str, List[str]], delete_trash: bool) -> None:
    if os.environ.get("CLEANUP_CONFIRM", "").lower() != "true":
        raise RuntimeError("Set CLEANUP_CONFIRM=true to apply cleanup.")

    archive_root = ROOT / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    for item in plan["archive"]:
        src = ROOT / item
        if not src.exists():
            continue
        dst = archive_root / item
        _move_path(src, dst)

    for item in plan["trash"]:
        src = ROOT / item
        if not src.exists():
            continue
        if delete_trash:
            if src.is_dir():
                shutil.rmtree(src)
            else:
                src.unlink()
        else:
            dst = archive_root / "trash" / item
            _move_path(src, dst)

    for move in plan["output_moves"]:
        src_str, dst_str = move.split(" -> ", 1)
        src = ROOT / src_str
        dst = ROOT / dst_str
        if not src.exists():
            continue
        _ensure_archive_dir(dst)
        if dst.exists():
            raise FileExistsError(f"Destination exists: {dst}")
        shutil.move(str(src), str(dst))


def main() -> None:
    args = parse_args()
    plan = build_plan()
    _print_plan(plan, args.max_list)

    if args.write_plan:
        out_path = Path(args.write_plan)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print(f"Wrote plan to {out_path}")

    if args.apply:
        apply_plan(plan, delete_trash=args.delete_trash)


if __name__ == "__main__":
    main()
