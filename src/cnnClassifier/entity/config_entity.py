from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    training_data: Path
    validation_data: Path
    trained_model_path: Path
    updated_base_model_path: Path   # 🔥 ADD THIS
    params_epochs: int
    params_batch_size: int
    params_image_size: list
    params_augmentation: bool