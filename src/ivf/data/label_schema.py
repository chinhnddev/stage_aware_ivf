"""
Label schema and parsing utilities for IVF embryo tasks.

Single source of truth for:
- StageLabel (cleavage/morula/blastocyst)
- QualityLabel (good/poor)
- Gardner grade parsing and mappings to quality and morphology targets

Hard rules: day metadata must not be used as model input and segmentation masks are disallowed.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class StageLabel(str, Enum):
    CLEAVAGE = "cleavage"
    MORULA = "morula"
    BLASTOCYST = "blastocyst"


class QualityLabel(str, Enum):
    GOOD = "good"
    POOR = "poor"


@dataclass(frozen=True)
class GardnerComponents:
    expansion: int
    icm: str
    te: str


# Expansion classes follow Gardner scale 1–6, IDs are zero-based to suit classifiers.
EXPANSION_CLASSES = [1, 2, 3, 4, 5, 6]
ICM_CLASSES = ["A", "B", "C"]
TE_CLASSES = ["A", "B", "C"]

EXPANSION_TO_ID: Dict[int, int] = {exp: idx for idx, exp in enumerate(EXPANSION_CLASSES)}
ICM_TO_ID: Dict[str, int] = {grade: idx for idx, grade in enumerate(ICM_CLASSES)}
TE_TO_ID: Dict[str, int] = {grade: idx for idx, grade in enumerate(TE_CLASSES)}

STAGE_CLASSES = [StageLabel.CLEAVAGE, StageLabel.MORULA, StageLabel.BLASTOCYST]
STAGE_TO_ID: Dict[StageLabel, int] = {stage: idx for idx, stage in enumerate(STAGE_CLASSES)}

QUALITY_CLASSES = [QualityLabel.POOR, QualityLabel.GOOD]
QUALITY_TO_ID: Dict[QualityLabel, int] = {label: idx for idx, label in enumerate(QUALITY_CLASSES)}

MISSING_GARDNER_TOKENS = {"", "0", "ND", "NA", "N/A"}
ICM_TE_NUMERIC_MAP = {1: "A", 2: "B", 3: "C"}
DEFAULT_Q_WEIGHTS = {"exp": 0.4, "icm": 0.3, "te": 0.3}

_GARDNER_PATTERN = re.compile(r"^\s*(?P<exp>[1-6])(?P<icm>[ABCabc123])(?P<te>[ABCabc123])\s*$")
_GARDNER_COMPONENT_PATTERN = re.compile(r"^\s*(?P<exp>[1-6])(?P<icm>[ABCabc123])?(?P<te>[ABCabc123])?\s*$")
_GARDNER_RANGE_PATTERN = re.compile(r"\d\s*-\s*\d")


def is_gardner_range_label(gardner: Optional[str]) -> bool:
    if gardner is None:
        return False
    return bool(_GARDNER_RANGE_PATTERN.search(str(gardner)))


def _is_missing_token(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    if isinstance(value, float) and value != value:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.upper() in MISSING_GARDNER_TOKENS


def normalize_gardner_exp(value) -> Optional[int]:
    if _is_missing_token(value):
        return None
    try:
        exp = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None
    if exp < 1 or exp > 6:
        return None
    return exp


def normalize_gardner_grade(value) -> Optional[str]:
    if _is_missing_token(value):
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C"}:
        return text
    try:
        num = int(float(text))
    except (ValueError, TypeError):
        return None
    return ICM_TE_NUMERIC_MAP.get(num)


def parse_gardner_components(gardner: Optional[str]) -> Optional[Tuple[int, Optional[str], Optional[str]]]:
    if gardner is None:
        return None
    if is_gardner_range_label(gardner):
        return None
    match = _GARDNER_COMPONENT_PATTERN.match(str(gardner))
    if not match:
        return None
    exp = normalize_gardner_exp(match.group("exp"))
    if exp is None:
        return None
    icm = normalize_gardner_grade(match.group("icm")) if match.group("icm") else None
    te = normalize_gardner_grade(match.group("te")) if match.group("te") else None
    return exp, icm, te


def parse_gardner(gardner: str) -> Optional[GardnerComponents]:
    """
    Parse a Gardner string like '4AA' or '3AB'.

    Returns:
        GardnerComponents if parsable, otherwise None.
    """
    if gardner is None:
        return None

    match = _GARDNER_PATTERN.match(str(gardner))
    if not match:
        return None

    expansion = int(match.group("exp"))
    icm = normalize_gardner_grade(match.group("icm"))
    te = normalize_gardner_grade(match.group("te"))
    if icm is None or te is None:
        return None
    return GardnerComponents(expansion, icm, te)


def map_gardner_to_quality(gardner: str) -> Optional[QualityLabel]:
    """
    Map a Gardner grade to a binary quality label.

    Rules:
        good -> 4AA, 5AA, and any 3-4AB (e.g., 3AB, 4AB)
        poor -> <=3CC (e.g., 1CC, 2CC, 3CC)
        others -> None (unknown/ignored)
    """
    components = parse_gardner(gardner)
    if components is None:
        return None

    exp, icm, te = components.expansion, components.icm, components.te

    if (exp in {4, 5} and icm == "A" and te == "A") or (exp in {3, 4} and icm == "A" and te == "B"):
        return QualityLabel.GOOD

    if exp <= 3 and icm == "C" and te == "C":
        return QualityLabel.POOR

    return None


UNSET = object()


def gardner_to_morphology_targets(
    gardner: Optional[str],
    exp_value=UNSET,
    icm_value=UNSET,
    te_value=UNSET,
) -> Dict[str, int]:
    """
    Convert Gardner grade into morphology class IDs for expansion, ICM, and TE.

    Raises:
        ValueError: if the Gardner string cannot be parsed or exp is missing.
    """
    if is_gardner_range_label(gardner):
        raise ValueError(f"Cannot parse Gardner range label: {gardner!r}")

    exp = normalize_gardner_exp(exp_value) if exp_value is not UNSET else None
    icm = normalize_gardner_grade(icm_value) if icm_value is not UNSET else None
    te = normalize_gardner_grade(te_value) if te_value is not UNSET else None

    components = parse_gardner_components(gardner)
    if exp is None and exp_value is UNSET and components is not None:
        exp = components[0]
    if icm is None and icm_value is UNSET and components is not None:
        icm = components[1]
    if te is None and te_value is UNSET and components is not None:
        te = components[2]

    if exp is None:
        raise ValueError(f"Cannot parse Gardner expansion: {gardner!r}")

    if exp < 3:
        icm = None
        te = None

    exp_mask = 1
    icm_mask = 1 if icm is not None and exp >= 3 else 0
    te_mask = 1 if te is not None and exp >= 3 else 0

    return {
        "exp": EXPANSION_TO_ID[exp],
        "icm": ICM_TO_ID[icm] if icm_mask else -1,
        "te": TE_TO_ID[te] if te_mask else -1,
        "exp_mask": exp_mask,
        "icm_mask": icm_mask,
        "te_mask": te_mask,
    }


def q_proxy_from_components(
    exp_value,
    icm_value,
    te_value,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """
    Compute a continuous quality proxy in [0, 1] from GT Gardner components.

    Returns None when any component is missing/invalid.
    """
    exp = normalize_gardner_exp(exp_value)
    icm = normalize_gardner_grade(icm_value)
    te = normalize_gardner_grade(te_value)
    if exp is None or icm is None or te is None:
        return None

    def _grade_score(value: str) -> float:
        return {"A": 1.0, "B": 0.5, "C": 0.0}[value]

    exp_norm = max(0.0, min(1.0, (exp - 1) / 5.0))
    icm_score = _grade_score(icm)
    te_score = _grade_score(te)

    weights = dict(DEFAULT_Q_WEIGHTS if weights is None else weights)
    w_exp = float(weights.get("exp", DEFAULT_Q_WEIGHTS["exp"]))
    w_icm = float(weights.get("icm", DEFAULT_Q_WEIGHTS["icm"]))
    w_te = float(weights.get("te", DEFAULT_Q_WEIGHTS["te"]))
    total = w_exp + w_icm + w_te
    if total <= 0:
        return None
    q = (w_exp * exp_norm + w_icm * icm_score + w_te * te_score) / total
    return max(0.0, min(1.0, float(q)))
