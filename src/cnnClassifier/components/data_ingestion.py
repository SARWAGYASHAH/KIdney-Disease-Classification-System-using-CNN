import os
import zipfile
import shutil
import random
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        logger.info("DataIngestion object created")

    def download_file(self) -> str:
        '''
        Fetch data from the url and store it in the zip file
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            if os.path.exists(zip_download_dir):
                logger.info(f"File already exists at {zip_download_dir}. Skipping download.")
                return zip_download_dir

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='

            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return zip_download_dir

        except Exception as e:
            logger.error("Error occurred during data download")
            raise e

    def extract_zip_file(self):
        """Extract zip file into data directory"""

        try:
            unzip_path = self.config.unzip_dir

            logger.info(f"Extracting zip file to: {unzip_path}")

            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info("Extraction completed successfully")

        except Exception as e:
            logger.error("Error occurred during extraction")
            raise e

    def flatten_folder(self):
        """Fix nested folder structure (optional use)"""

        try:
            base_path = self.config.unzip_dir

            logger.info(f"Checking folder structure in: {base_path}")

            dirs = os.listdir(base_path)

            if len(dirs) == 1:
                inner_path = os.path.join(base_path, dirs[0])

                if os.path.isdir(inner_path):
                    logger.info("Nested folder detected. Flattening structure")

                    inner_items = os.listdir(inner_path)

                    for item in inner_items:
                        src = os.path.join(inner_path, item)
                        dest = os.path.join(base_path, item)
                        shutil.move(src, dest)

                    os.rmdir(inner_path)

                    logger.info("Folder structure flattened successfully")
                else:
                    logger.info("No nested directory found")
            else:
                logger.info("Folder structure already correct")

        except Exception as e:
            logger.error("Error occurred while flattening folder structure")
            raise e

    def split_data(self):
        try:
            base_path = self.config.unzip_dir

            dataset_path = os.path.join(
              base_path,
              "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
              "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

            train_dir = os.path.join(base_path, "train")
            valid_dir = os.path.join(base_path, "valid")

            if os.path.exists(train_dir) and os.path.exists(valid_dir):
                logger.info("Data already split. Skipping splitting step.")
                return

            logger.info("Starting dataset splitting")

            split_ratio = 0.8

            for class_name in os.listdir(dataset_path):

                if class_name in ["train", "valid", "data.zip", "kidneyData.csv"]:
                    continue

                class_path = os.path.join(dataset_path, class_name)

                if not os.path.isdir(class_path):
                    continue

                images = os.listdir(class_path)

                if len(images) == 0:
                    logger.warning(f"No images found in {class_name}")
                    continue

                random.shuffle(images)

                split_index = int(len(images) * split_ratio)

                train_images = images[:split_index]
                valid_images = images[split_index:]

                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

                for img in train_images:
                    shutil.copy(
                        os.path.join(class_path, img),
                        os.path.join(train_dir, class_name, img)
                    )

                for img in valid_images:
                    shutil.copy(
                        os.path.join(class_path, img),
                        os.path.join(valid_dir, class_name, img)
                    )

                logger.info(f"{class_name}: {len(train_images)} train, {len(valid_images)} valid")

            logger.info("Dataset splitting completed successfully")

        except Exception as e:
            logger.error("Error occurred during dataset splitting")
            raise e