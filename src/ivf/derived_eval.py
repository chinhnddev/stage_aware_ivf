"""
Derived morphology-based evaluation utilities for Phase-4 q_score.

These helpers are evaluation-only and must not be used for training.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from ivf.data.label_schema import normalize_gardner_exp, normalize_gardner_grade, parse_gardner_components


_ICM_TE_TO_NUM = {"A": 1, "B": 2, "C": 3}


def _to_numeric_grade(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            num = int(float(value))
        except (TypeError, ValueError):
            return None
        return num if num in {1, 2, 3} else None
    text = str(value).strip().upper()
    if text in _ICM_TE_TO_NUM:
        return _ICM_TE_TO_NUM[text]
    try:
        num = int(float(text))
    except (TypeError, ValueError):
        return None
    return num if num in {1, 2, 3} else None


def _normalize_rule_set(values: Iterable) -> set[int]:
    normalized = set()
    for value in values:
        num = _to_numeric_grade(value)
        if num is not None:
            normalized.add(num)
    return normalized


def parse_morphology_components(
    exp,
    icm,
    te,
    grade=None,
    gardner=None,
) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    """
    Parse morphology components from exp/icm/te or fallback to Gardner parsing.

    Returns:
        (exp, icm, te, source) where icm/te are numeric {1,2,3} and source is a string.
    """
    exp_val = normalize_gardner_exp(exp)
    icm_val = _to_numeric_grade(normalize_gardner_grade(icm))
    te_val = _to_numeric_grade(normalize_gardner_grade(te))
    if exp_val is not None:
        return exp_val, icm_val, te_val, "exp_icm_te"

    raw = gardner if gardner is not None else grade
    components = parse_gardner_components(raw)
    if components is None:
        return None, None, None, "missing"
    exp_val, icm_letter, te_letter = components
    icm_val = _to_numeric_grade(icm_letter)
    te_val = _to_numeric_grade(te_letter)
    return exp_val, icm_val, te_val, "gardner_parse"


def derive_good_poor_from_morph(
    exp,
    icm,
    te,
    rule_cfg,
    grade=None,
    gardner=None,
    return_components: bool = False,
):
    """
    Derive morphology-based good/poor label for evaluation only.

    Rule (default):
        good if exp >= 3 and icm in {A,B} and te in {A,B}, else poor.
    """
    exp_min = int(getattr(rule_cfg, "exp_min", 3)) if rule_cfg is not None else 3
    icm_good = _normalize_rule_set(getattr(rule_cfg, "icm_good", [1, 2]) if rule_cfg is not None else [1, 2])
    te_good = _normalize_rule_set(getattr(rule_cfg, "te_good", [1, 2]) if rule_cfg is not None else [1, 2])

    exp_val, icm_val, te_val, source = parse_morphology_components(exp, icm, te, grade=grade, gardner=gardner)
    if exp_val is None:
        label = None
    elif exp_val < exp_min:
        label = 0
    else:
        if icm_val is None or te_val is None:
            label = None
        else:
            label = 1 if (icm_val in icm_good and te_val in te_good) else 0

    if return_components:
        return label, exp_val, icm_val, te_val, source
    return label


def compute_derived_binary_metrics(scores: Iterable[float], labels: Iterable[int], threshold: Optional[float] = None) -> dict:
    scores_list = list(scores)
    labels_list = list(labels)
    result = {"auroc": None, "auprc": None}
    if not scores_list or not labels_list:
        return result
    scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
    try:
        result["auroc"] = float(BinaryAUROC()(scores_tensor, labels_tensor))
    except ValueError:
        result["auroc"] = None
    try:
        result["auprc"] = float(BinaryAveragePrecision()(scores_tensor, labels_tensor))
    except ValueError:
        result["auprc"] = None

    if threshold is None:
        return result

    preds = [1 if score >= threshold else 0 for score in scores_list]
    tp = sum(1 for p, y in zip(preds, labels_list) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels_list) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels_list) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels_list) if p == 0 and y == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(labels_list) if labels_list else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (recall + tnr)
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    result.update(
        {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "acc": acc,
            "bal_acc": bal_acc,
        }
    )
    return result


def tune_threshold_on_val(
    scores_val: Iterable[float],
    labels_val: Iterable[int],
    objective: str = "f1",
    grid: Optional[Iterable[float]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    scores = list(scores_val)
    labels = list(labels_val)
    if not scores or not labels:
        return None, None
    if len(set(labels)) < 2:
        return None, None

    if grid is None:
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return None, None
        if 0.0 <= min_score and max_score <= 1.0:
            grid = [i / 100 for i in range(0, 101)]
        else:
            step = (max_score - min_score) / 100
            grid = [min_score + i * step for i in range(0, 101)]

    best_thresh = None
    best_score = None
    for thresh in grid:
        metrics = compute_derived_binary_metrics(scores, labels, threshold=thresh)
        score = metrics.get("f1") if objective == "f1" else metrics.get("bal_acc")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_thresh = float(thresh)
    return best_thresh, best_score
