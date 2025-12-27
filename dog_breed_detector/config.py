from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataConfig:
    data_dir: str
    train_images: str
    train_labels: str
    test_size: float
    num_workers: int


@dataclass
class PreprocessingConfig:
    channel_mean: List[float]
    channel_std: List[float]
    resize: int
    image_size: int
    random_horizontal_flip: float
    random_rotation: float


@dataclass
class ModelConfig:
    name: str
    weights: str
    num_classes: int
    seed: int
    criterion: str


@dataclass
class OptimConfig:
    learning_rate: float
    momentum: float
    step_size: int
    gamma: float


@dataclass
class TrainConfig:
    batch_size: int
    valid_batch_size: int
    epochs: int
    shuffle: bool
    device: str
    early_stopping_patience: int
    gradient_clip_val: float
    log_interval: int
    checkpoint_dir: str
    save_top_k: int
    visualize_samples: bool


@dataclass
class LoggingConfig:
    enabled: bool
    experiment_name: str
    tracking_uri: str
    run_name: str


@dataclass
class CheckpointConfig:
    enabled: bool
    save_dir: str
    monitor: str
    mode: str


@dataclass
class ExportOnnxConfig:
    model_path: str
    onnx_path: str
    preprocessing_path: str
    postprocessing_path: str
    test_data_dir: str
    opset_version: int
    dynamic_batch: bool
    export_preprocessing: bool
    export_postprocessing: bool


@dataclass
class ExportTensorRTConfig:
    onnx_path: str
    engine_dir: str
    precision: str
    workspace_gb: int
    max_batch_size: int
    dynamic_shapes: Dict[str, List[List[int]]]
    validate: bool
    benchmark: bool


@dataclass
class AppConfig:
    seed: int
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig