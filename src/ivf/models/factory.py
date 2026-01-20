"""
Model factory for IVF experiments.
"""

from typing import Optional

from ivf.models.morph_backbone import MorphologyBackbone
from ivf.models.morph_joint import JointMorphNet
from ivf.models.morph_paper import PaperMorphNet
from ivf.models.multitask import MultiTaskEmbryoNet


def build_model_from_config(cfg, phase: Optional[str] = None):
    model_cfg = cfg.model
    encoder_cfg = model_cfg.encoder
    model_name = str(getattr(model_cfg, "name", "multitask")).lower()
    if model_name in {"morph_paper", "paper_morph", "paper"}:
        if phase is not None and phase != "morph":
            raise ValueError(f"Model {model_name} is only supported for phase=morph.")
        morph_cfg = getattr(cfg.training, "morph", None)
        backbone = str(getattr(morph_cfg, "paper_backbone", "resnet50")) if morph_cfg is not None else "resnet50"
        pretrained = bool(getattr(morph_cfg, "paper_pretrained", True)) if morph_cfg is not None else True
        return PaperMorphNet(
            backbone=backbone,  # type: ignore[arg-type]
            pretrained=pretrained,
            head_hidden_dim=int(getattr(model_cfg, "head_hidden_dim", 0)),
        )
    if model_name in {"joint_morph", "morph_joint", "jointmorphnet"}:
        if phase is not None and phase != "morph":
            raise ValueError(f"Model {model_name} is only supported for phase=morph.")
        morph_cfg = getattr(cfg.training, "morph", None)
        exp_max = int(getattr(morph_cfg, "exp_max", 5)) if morph_cfg is not None else 5
        encoder = MorphologyBackbone(
            in_channels=encoder_cfg.in_channels,
            dims=encoder_cfg.dims,
            feature_dim=encoder_cfg.feature_dim,
            width_mult=float(getattr(model_cfg, "width_mult", 1.0)),
            depth_mult=float(getattr(model_cfg, "depth_mult", 1.0)),
            fusion_mode=str(getattr(model_cfg, "fusion_mode", "concat")),
            attention_type=str(getattr(model_cfg, "attention_type", "eca")),
            attention_kernel=int(getattr(model_cfg, "attention_kernel", 3)),
        )
        return JointMorphNet(
            encoder=encoder,
            feature_dim=encoder.feature_dim,
            exp_num_classes=exp_max,
            icm_num_classes=3,
            te_num_classes=3,
            head_hidden_dim=int(getattr(model_cfg, "head_hidden_dim", 0)),
        )

    encoder = MorphologyBackbone(
        in_channels=encoder_cfg.in_channels,
        dims=encoder_cfg.dims,
        feature_dim=encoder_cfg.feature_dim,
        width_mult=float(getattr(model_cfg, "width_mult", 1.0)),
        depth_mult=float(getattr(model_cfg, "depth_mult", 1.0)),
        fusion_mode=str(getattr(model_cfg, "fusion_mode", "concat")),
        attention_type=str(getattr(model_cfg, "attention_type", "eca")),
        attention_kernel=int(getattr(model_cfg, "attention_kernel", 3)),
    )

    return MultiTaskEmbryoNet(
        encoder=encoder,
        feature_dim=encoder.feature_dim,
        quality_mode=model_cfg.heads.quality_mode,
        quality_conditioning=getattr(model_cfg.heads, "quality_conditioning", "morph+stage"),
    )
