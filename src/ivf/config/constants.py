"""
Project-wide constants for IVF tasks and guardrails.
"""

TASKS = {
    "blastocyst_morphology": {
        "description": "Multi-head classification for expansion (EXP), ICM, and TE grades.",
        "heads": ["exp", "icm", "te"],
    },
    "stage_classification": {
        "description": "Stage classification across cleavage/morula/blastocyst.",
        "labels": ["cleavage", "morula", "blastocyst"],
    },
    "quality_prediction": {
        "description": "Binary quality prediction (good/poor) derived from Gardner grades.",
        "labels": ["good", "poor"],
    },
}

# Guardrail configurations
DAY_FEATURE_KEYS = {"day", "day_label", "day_str"}
SEGMENTATION_KEYS = {"mask", "segmentation", "seg_mask", "roi_mask", "binary_mask"}
