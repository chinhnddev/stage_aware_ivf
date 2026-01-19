"""
Morphology utilities for joint EXP/ICM/TE evaluation.
"""

from typing import Dict, Optional

import numpy as np


def _score_config(config) -> Dict[str, float]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return {
        "weights": getattr(config, "weights", None),
        "missing_strategy": getattr(config, "missing_strategy", None),
        "neutral_icm_score": getattr(config, "neutral_icm_score", None),
        "neutral_te_score": getattr(config, "neutral_te_score", None),
        "confidence_lambda": getattr(config, "confidence_lambda", None),
    }


def morph_score(
    exp_probs,
    icm_probs,
    te_probs,
    valid_icm: bool,
    valid_te: bool,
    config: Optional[object] = None,
) -> float:
    cfg = _score_config(config)
    weights = cfg.get("weights") or {"exp": 1.0, "icm": 1.0, "te": 1.0}
    w_exp = float(weights.get("exp", 1.0))
    w_icm = float(weights.get("icm", 1.0))
    w_te = float(weights.get("te", 1.0))
    missing_strategy = str(cfg.get("missing_strategy") or "neutral").lower()
    neutral_icm = float(cfg.get("neutral_icm_score") or 2.0)
    neutral_te = float(cfg.get("neutral_te_score") or 2.0)
    conf_lambda = float(cfg.get("confidence_lambda") or 0.0)

    exp_probs = np.asarray(exp_probs, dtype=float)
    icm_probs = np.asarray(icm_probs, dtype=float)
    te_probs = np.asarray(te_probs, dtype=float)

    exp_scores = np.arange(1, exp_probs.size + 1, dtype=float)
    icm_scores = np.array([3.0, 2.0, 1.0], dtype=float)
    te_scores = np.array([3.0, 2.0, 1.0], dtype=float)

    e_exp = float((exp_probs * exp_scores).sum()) if exp_probs.size else 0.0
    e_icm = float((icm_probs * icm_scores[: icm_probs.size]).sum()) if icm_probs.size else 0.0
    e_te = float((te_probs * te_scores[: te_probs.size]).sum()) if te_probs.size else 0.0

    if not valid_icm:
        if missing_strategy == "neutral":
            e_icm = neutral_icm
        elif missing_strategy == "skip":
            w_icm = 0.0
    if not valid_te:
        if missing_strategy == "neutral":
            e_te = neutral_te
        elif missing_strategy == "skip":
            w_te = 0.0

    q = w_exp * e_exp + w_icm * e_icm + w_te * e_te

    if conf_lambda > 0:
        q -= conf_lambda * (1.0 - float(exp_probs.max())) if exp_probs.size else 0.0
        if valid_icm and icm_probs.size:
            q -= conf_lambda * (1.0 - float(icm_probs.max()))
        if valid_te and te_probs.size:
            q -= conf_lambda * (1.0 - float(te_probs.max()))

    return float(q)
