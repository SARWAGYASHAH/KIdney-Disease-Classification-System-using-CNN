from dataclasses import dataclass
from pathlib import Path

from kidney_disease_classifier import logger
from kidney_disease_classifier.utils.common import create_directories, read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    source_url: str | None
    source_data_path: Path | None
    google_drive_data_path: Path | None
    train_data_path: Path
    valid_data_path: Path
    test_data_path: Path
    classes: list[str]
    image_size: list[int]
    batch_size: int
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    random_state: int
    supported_extensions: set[str]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    test_data_path: Path
    evaluation_dir: Path
    scores_file: Path
    confusion_matrix_path: Path
    image_size: list[int]
    batch_size: int
    classes: list[str]


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    image_size: list[int]
    classes: list[str]


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = DEFAULT_CONFIG_PATH,
        params_filepath: Path = DEFAULT_PARAMS_PATH,
    ) -> None:
        logger.info("ConfigurationManager initialization started")
        self.config_filepath = config_filepath
        self.params_filepath = params_filepath
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories(
            [
                self._resolve_path(self.config.artifacts_root),
                self._resolve_path(self.config.data_ingestion_root),
                self._resolve_path(self.config.raw_data_path),
                self._resolve_path(self.config.evaluation_dir),
                PROJECT_ROOT / "logs",
            ]
        )
        logger.info("ConfigurationManager initialized successfully")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Preparing data ingestion configuration")
        split_ratio = self.config.split_ratio
        return DataIngestionConfig(
            raw_data_path=self._resolve_path(self.config.raw_data_path),
            source_url=self.config.get("source_url"),
            source_data_path=self._optional_path(self.config.source_data_path),
            google_drive_data_path=self._optional_path(self.config.google_drive_data_path),
            train_data_path=self._resolve_path(self.config.train_data_path),
            valid_data_path=self._resolve_path(self.config.valid_data_path),
            test_data_path=self._resolve_path(self.config.test_data_path),
            classes=list(self.config.classes),
            image_size=list(self.params.IMAGE_SIZE),
            batch_size=int(self.params.BATCH_SIZE),
            train_ratio=float(split_ratio.train),
            valid_ratio=float(split_ratio.valid),
            test_ratio=float(split_ratio.test),
            random_state=int(self.config.random_state),
            supported_extensions={extension.lower() for extension in self.config.supported_extensions},
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        logger.info("Preparing model evaluation configuration")
        return ModelEvaluationConfig(
            model_path=self._resolve_path(self.config.model_path),
            test_data_path=self._resolve_path(self.config.test_data_path),
            evaluation_dir=self._resolve_path(self.config.evaluation_dir),
            scores_file=self._resolve_path(self.config.scores_file),
            confusion_matrix_path=self._resolve_path(self.config.confusion_matrix_path),
            image_size=list(self.params.IMAGE_SIZE),
            batch_size=int(self.params.BATCH_SIZE),
            classes=list(self.config.classes),
        )

    def get_prediction_config(self) -> PredictionConfig:
        logger.info("Preparing prediction configuration")
        return PredictionConfig(
            model_path=self._resolve_path(self.config.model_path),
            image_size=list(self.params.IMAGE_SIZE),
            classes=list(self.config.classes),
        )

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _optional_path(self, path_value):
        if path_value in (None, "", "null"):
            return None
        return self._resolve_path(path_value)
