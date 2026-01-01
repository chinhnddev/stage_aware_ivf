"""
Structured config schema for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EncoderConfig:
    in_channels: int = 3
    dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    feature_dim: int = 256
    weights_path: Optional[str] = None


@dataclass
class HeadsConfig:
    quality_mode: str = "concat"


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    encoder_config: Optional[str] = None
    heads_config: Optional[str] = None


@dataclass
class DataConfig:
    blastocyst_config: str = "configs/data/blastocyst.yaml"
    humanembryo2_config: str = "configs/data/humanembryo2.yaml"
    quality_config: str = "configs/data/quality_public.yaml"
    hungvuong_config: str = "configs/data/hungvuong.yaml"
    splits_base_dir: str = "data/processed/splits"
    include_meta_day_default: bool = True


@dataclass
class TransformConfig:
    image_size: int = 256
    morph: str = "light"
    stage: str = "medium"
    joint: str = "light"
    quality: str = "light"
    normalize: bool = False
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class FreezeConfig:
    stage_start_ratio: float = 0.8
    stage_schedule: List[List[float]] = field(default_factory=lambda: [[0, 0.8], [5, 0.5], [10, 0.0]])


@dataclass
class TrainingConfig:
    lr: float = 0.001
    weight_decay: float = 0.0001
    epochs: Dict[str, int] = field(default_factory=lambda: {"morph": 20, "stage": 15, "joint": 10, "quality": 10})
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"morph": 1.0, "stage": 1.0, "quality": 1.0})
    freeze: FreezeConfig = field(default_factory=FreezeConfig)


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    csv: bool = False


@dataclass
class OutputConfig:
    checkpoints_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    reports_dir: str = "outputs/reports"


@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    base_config: Optional[str] = None
