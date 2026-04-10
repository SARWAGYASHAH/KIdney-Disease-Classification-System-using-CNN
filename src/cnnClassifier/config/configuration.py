from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        try:
            logger.info("Reading configuration files")

            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)

            logger.info("Configuration files loaded successfully")

            logger.info(f"Creating artifacts root directory at: {self.config.artifacts_root}")
            create_directories([self.config.artifacts_root])

        except Exception as e:
            logger.error("Error while initializing ConfigurationManager")
            raise e


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logger.info("Preparing Data Ingestion Configuration")

            config = self.config.data_ingestion

            logger.info(f"Creating data ingestion directory at: {config.root_dir}")
            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir
            )

            logger.info("Data Ingestion Configuration created successfully")

            return data_ingestion_config

        except Exception as e:
            logger.error("Error while creating Data Ingestion Configuration")
            raise e