"""
Adapter for the Kaggle Human blastocyst dataset (morphology learning).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Tuple, Union

import pandas as pd

from ivf.data.datasets import BaseImageDataset
from ivf.data.label_schema import (
    MISSING_GARDNER_TOKENS,
    UNSET,
    gardner_to_morphology_targets,
    is_gardner_range_label,
    map_gardner_to_quality,
    normalize_gardner_exp,
    normalize_gardner_grade,
    parse_gardner_components,
)
from ivf.utils.logging import get_logger


@dataclass
class GardnerParseStats:
    total: int = 0
    kept: int = 0
    dropped_range_label: int = 0
    missing_exp: int = 0
    invalid_exp: int = 0
    exp_lt3: int = 0
    missing_icm: int = 0
    invalid_icm: int = 0
    missing_te: int = 0
    invalid_te: int = 0

    def as_dict(self):
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped_range_label": self.dropped_range_label,
            "missing_exp": self.missing_exp,
            "invalid_exp": self.invalid_exp,
            "exp_lt3": self.exp_lt3,
            "missing_icm": self.missing_icm,
            "invalid_icm": self.invalid_icm,
            "missing_te": self.missing_te,
            "invalid_te": self.invalid_te,
        }


def _pick_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _is_missing_token(value) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper() in MISSING_GARDNER_TOKENS


def _update_icm_te_stats(raw_value, normalized_value, stats: GardnerParseStats, field: str) -> None:
    if _is_missing_token(raw_value):
        if field == "icm":
            stats.missing_icm += 1
        else:
            stats.missing_te += 1
        return
    if normalized_value is None:
        if field == "icm":
            stats.invalid_icm += 1
        else:
            stats.invalid_te += 1


def _log_parse_stats(stats: GardnerParseStats, context: str) -> None:
    logger = get_logger("ivf")
    logger.info("Blastocyst Gardner parsing stats (%s): %s", context, stats.as_dict())


def load_blastocyst_records(
    root_dir: Path,
    csv_path: Path,
    image_col: str,
    grade_col: str,
    id_col: str,
    day_col: Optional[str] = None,
    return_stats: bool = False,
    log_stats: bool = True,
) -> Union[List[MutableMapping], Tuple[List[MutableMapping], GardnerParseStats]]:
    df = pd.read_csv(csv_path)
    exp_col = _pick_column(df, ["exp", "EXP", "expansion"])
    icm_col = _pick_column(df, ["icm", "ICM"])
    te_col = _pick_column(df, ["te", "TE"])
    stats = GardnerParseStats()
    records: List[MutableMapping] = []
    for _, row in df.iterrows():
        stats.total += 1
        grade = row.get(grade_col) if grade_col in df.columns else None
        if is_gardner_range_label(grade):
            stats.dropped_range_label += 1
            continue

        exp_raw = row.get(exp_col) if exp_col else None
        icm_raw = row.get(icm_col) if icm_col else None
        te_raw = row.get(te_col) if te_col else None
        components = parse_gardner_components(grade)

        exp = normalize_gardner_exp(exp_raw)
        if exp is None and exp_col is None and components is not None:
            exp = components[0]
        if exp is None:
            if _is_missing_token(exp_raw) or (exp_raw is None and components is None):
                stats.missing_exp += 1
            else:
                stats.invalid_exp += 1
            continue
        if exp < 3:
            stats.exp_lt3 += 1
        if exp >= 3:
            icm_norm = normalize_gardner_grade(icm_raw) if icm_col else (components[1] if components else None)
            te_norm = normalize_gardner_grade(te_raw) if te_col else (components[2] if components else None)
            _update_icm_te_stats(icm_raw if icm_col else icm_norm, icm_norm, stats, "icm")
            _update_icm_te_stats(te_raw if te_col else te_norm, te_norm, stats, "te")

        try:
            targets = gardner_to_morphology_targets(
                grade,
                exp_value=exp,
                icm_value=icm_raw if icm_col else UNSET,
                te_value=te_raw if te_col else UNSET,
            )
        except ValueError:
            stats.invalid_exp += 1
            continue  # skip unknown/invalid grades

        quality = map_gardner_to_quality(grade)
        icm_meta = normalize_gardner_grade(icm_raw) if icm_col else (components[1] if components else None)
        te_meta = normalize_gardner_grade(te_raw) if te_col else (components[2] if components else None)
        if exp < 3:
            icm_meta = None
            te_meta = None
        meta = {
            "id": row.get(id_col),
            "dataset": "blastocyst_kaggle",
            "grade": grade,
            "exp": exp,
            "icm": icm_meta,
            "te": te_meta,
        }
        if quality:
            meta["quality"] = quality.value
        if day_col and day_col in df.columns and pd.notna(row.get(day_col)):
            meta["day"] = row.get(day_col)

        record: MutableMapping = {
            "image_path": str(Path(root_dir) / str(row.get(image_col))),
            "targets": targets,
            "meta": meta,
        }
        records.append(record)
        stats.kept += 1
    if log_stats:
        _log_parse_stats(stats, context="load")
    if return_stats:
        return records, stats
    return records


def records_to_dataframe(records: Iterable[MutableMapping]) -> pd.DataFrame:
    rows = []
    for rec in records:
        meta = rec.get("meta", {})
        rows.append(
            {
                "image_path": rec.get("image_path"),
                "id": meta.get("id"),
                "grade": meta.get("grade"),
                "exp": meta.get("exp"),
                "icm": meta.get("icm"),
                "te": meta.get("te"),
                "quality": meta.get("quality"),
                "day": meta.get("day"),
                "dataset": meta.get("dataset"),
            }
        )
    return pd.DataFrame(rows)


class BlastocystKaggleDataset(BaseImageDataset):
    """
    Morphology dataset using Gardner grades for EXP/ICM/TE targets.
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        image_col: str,
        grade_col: str,
        id_col: str,
        transform=None,
        include_meta_day: bool = True,
        day_col: Optional[str] = None,
    ) -> None:
        records = load_blastocyst_records(
            Path(root_dir),
            Path(csv_path),
            image_col=image_col,
            grade_col=grade_col,
            id_col=id_col,
            day_col=day_col,
            return_stats=True,
            log_stats=False,
        )
        records, stats = records
        _log_parse_stats(stats, context="dataset_init")
        super().__init__(records, transform=transform, include_meta_day=include_meta_day)
