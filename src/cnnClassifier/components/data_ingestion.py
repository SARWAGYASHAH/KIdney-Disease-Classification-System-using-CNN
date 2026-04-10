import os
import zipfile
import shutil
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
        Fetch data from the url and store it in the zip file then later in the pipeline got unzipped
        '''
    
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
    
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
    
            # ✅ ADD THIS CHECK
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
        """Fix nested folder structure"""

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