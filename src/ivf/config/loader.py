"""
Config loader with OmegaConf and structured validation.
"""

from typing import Any, Dict

from omegaconf import OmegaConf

from ivf.config.schema import ExperimentConfig


def _load_yaml(path: str):
    return OmegaConf.load(path)


def _merge_structured(raw_cfg) -> ExperimentConfig:
    base = OmegaConf.structured(ExperimentConfig)
    OmegaConf.set_struct(base, True)
    merged = OmegaConf.merge(base, raw_cfg)
    OmegaConf.set_struct(merged, True)
    return merged


def _load_with_base(config_path: str):
    raw = _load_yaml(config_path)
    base_path = raw.get("base_config")
    if base_path:
        base = _load_yaml(base_path)
        raw = OmegaConf.merge(base, raw)
    return raw


def load_experiment_config(config_path: str):
    raw = _load_with_base(config_path)

    model_cfg = raw.get("model", {})
    encoder_cfg_path = model_cfg.get("encoder_config")
    heads_cfg_path = model_cfg.get("heads_config")

    if encoder_cfg_path:
        encoder_cfg = _load_yaml(encoder_cfg_path)
        model_cfg.encoder = OmegaConf.merge(encoder_cfg, model_cfg.get("encoder", {}))
    if heads_cfg_path:
        heads_cfg = _load_yaml(heads_cfg_path)
        model_cfg.heads = OmegaConf.merge(heads_cfg, model_cfg.get("heads", {}))

    raw.model = model_cfg
    cfg = _merge_structured(raw)
    return cfg


def resolve_config_dict(cfg) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)
