"""
Model factory for IVF experiments.
"""

from typing import Optional

from ivf.models.encoder import ConvNeXtMini
from ivf.models.morph_joint import JointMorphNet
from ivf.models.multitask import MultiTaskEmbryoNet


def build_model_from_config(cfg, phase: Optional[str] = None):
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    encoder = ConvNeXtMini(
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        width_mult=float(getattr(model_cfg, "width_mult", 1.0)),
        depth_mult=float(getattr(model_cfg, "depth_mult", 1.0)),
        weights_path=encoder_cfg.weights_path,
    )

    model_name = str(getattr(model_cfg, "name", "multitask")).lower()
    if model_name in {"joint_morph", "morph_joint", "jointmorphnet"}:
        if phase is not None and phase != "morph":
            raise ValueError(f"Model {model_name} is only supported for phase=morph.")
        morph_cfg = getattr(cfg.training, "morph", None)
        exp_max = int(getattr(morph_cfg, "exp_max", 6)) if morph_cfg is not None else 6
        return JointMorphNet(
            encoder=encoder,
            feature_dim=encoder.feature_dim,
            exp_num_classes=exp_max,
            icm_num_classes=3,
            te_num_classes=3,
        )

    return MultiTaskEmbryoNet(
        encoder=encoder,
        feature_dim=encoder.feature_dim,
        quality_mode=model_cfg.heads.quality_mode,
        quality_conditioning=getattr(model_cfg.heads, "quality_conditioning", "morph+stage"),
    )
