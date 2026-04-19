import random
import shutil
import zipfile
from pathlib import Path

import gdown

from kidney_disease_classifier import logger
from kidney_disease_classifier.config.configuration import DataIngestionConfig
from kidney_disease_classifier.utils.common import create_directories


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        logger.info("DataIngestion initialized with source path: %s", self.config.source_data_path)

    def run(self) -> None:
        logger.info("Data ingestion stage started")
        logger.info(
            "Using paths - raw: %s, train: %s, valid: %s, test: %s",
            self.config.raw_data_path,
            self.config.train_data_path,
            self.config.valid_data_path,
            self.config.test_data_path,
        )

        try:
            if self._split_dataset_exists():
                logger.info("Split dataset already exists. Skipping data ingestion stage.")
                return

            source_path = self._resolve_source_path()
            dataset_root = self._prepare_source(source_path)
            class_root = self._locate_class_root(dataset_root)
            self._split_dataset(class_root)
            logger.info("Data ingestion stage completed successfully")
        except Exception as error:
            logger.exception("Data ingestion stage failed: %s", error)
            raise

    def _resolve_source_path(self) -> Path:
        candidate_paths = [self.config.source_data_path, self.config.google_drive_data_path]
        for path in candidate_paths:
            if path and path.exists():
                logger.info("Resolved dataset source path: %s", path)
                return path

        if self.config.source_url:
            return self._download_from_google_drive(self.config.source_url)

        raise FileNotFoundError(
            "Dataset source not found. Set source_data_path, google_drive_data_path, or source_url in config/config.yaml."
        )

    def _download_from_google_drive(self, source_url: str) -> Path:
        create_directories([self.config.raw_data_path])

        file_id = self._extract_google_drive_file_id(source_url)
        download_path = self.config.raw_data_path / "dataset.zip"

        if download_path.exists():
            logger.info("Google Drive dataset already downloaded at %s", download_path)
            return download_path

        logger.info("Downloading dataset from Google Drive URL: %s", source_url)
        gdown.download(id=file_id, output=str(download_path), quiet=False)
        logger.info("Dataset downloaded successfully to %s", download_path)
        return download_path

    def _extract_google_drive_file_id(self, source_url: str) -> str:
        url_parts = source_url.split("/")
        if "file" in url_parts and "d" in url_parts:
            file_id_index = url_parts.index("d") + 1
            if file_id_index < len(url_parts):
                return url_parts[file_id_index]

        raise ValueError(f"Unsupported Google Drive URL format: {source_url}")

    def _prepare_source(self, source_path: Path) -> Path:
        if source_path.is_file() and source_path.suffix.lower() == ".zip":
            extract_dir = self.config.raw_data_path / source_path.stem
            if not extract_dir.exists():
                logger.info("Extracting zip dataset from %s to %s", source_path, extract_dir)
                create_directories([extract_dir])
                with zipfile.ZipFile(source_path, "r") as zip_file:
                    zip_file.extractall(extract_dir)
            else:
                logger.info("Zip dataset already extracted at %s", extract_dir)
            return extract_dir

        logger.info("Using directory dataset source at %s", source_path)
        return source_path

    def _locate_class_root(self, dataset_root: Path) -> Path:
        if self._contains_expected_class_dirs(dataset_root):
            logger.info("Dataset root contains expected class folders: %s", dataset_root)
            return dataset_root

        for directory in dataset_root.rglob("*"):
            if directory.is_dir() and self._contains_expected_class_dirs(directory):
                logger.info("Located nested dataset root at %s", directory)
                return directory

        raise FileNotFoundError(
            f"Could not locate class folders {self.config.classes} under source path {dataset_root}."
        )

    def _contains_expected_class_dirs(self, path: Path) -> bool:
        available_dirs = {item.name for item in path.iterdir() if item.is_dir()} if path.exists() else set()
        return set(self.config.classes).issubset(available_dirs)

    def _split_dataset(self, class_root: Path) -> None:
        self._reset_split_directories()

        for class_name in self.config.classes:
            source_class_dir = class_root / class_name
            image_files = [
                file_path
                for file_path in source_class_dir.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in self.config.supported_extensions
            ]

            if not image_files:
                raise FileNotFoundError(f"No supported images found for class '{class_name}' in {source_class_dir}.")

            random.Random(self.config.random_state).shuffle(image_files)
            total_images = len(image_files)
            train_end = int(total_images * self.config.train_ratio)
            valid_end = train_end + int(total_images * self.config.valid_ratio)

            split_map = {
                self.config.train_data_path / class_name: image_files[:train_end],
                self.config.valid_data_path / class_name: image_files[train_end:valid_end],
                self.config.test_data_path / class_name: image_files[valid_end:],
            }

            for target_dir, files in split_map.items():
                create_directories([target_dir])
                for file_path in files:
                    shutil.copy2(file_path, target_dir / file_path.name)

            logger.info(
                "Class '%s' split counts - train: %s, valid: %s, test: %s",
                class_name,
                len(split_map[self.config.train_data_path / class_name]),
                len(split_map[self.config.valid_data_path / class_name]),
                len(split_map[self.config.test_data_path / class_name]),
            )

    def _reset_split_directories(self) -> None:
        for split_dir in [self.config.train_data_path, self.config.valid_data_path, self.config.test_data_path]:
            if split_dir.exists():
                logger.info("Removing existing split directory: %s", split_dir)
                shutil.rmtree(split_dir)
            create_directories([split_dir])

    def _split_dataset_exists(self) -> bool:
        split_dirs = [self.config.train_data_path, self.config.valid_data_path, self.config.test_data_path]
        for split_dir in split_dirs:
            if not split_dir.exists():
                return False
            if not any(split_dir.rglob("*.*")):
                return False
        return True
