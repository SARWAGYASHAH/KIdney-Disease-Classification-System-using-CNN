from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger
import os

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    '''this class handles downloading, extracting and splitting data'''

    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("Initializing Configuration Manager")
            config = ConfigurationManager()

            logger.info("Fetching Data Ingestion Configuration")
            data_ingestion_config = config.get_data_ingestion_config()

            logger.info("Initializing Data Ingestion Component")
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # PATHS
            unzip_dir = data_ingestion_config.unzip_dir

            # DOWNLOAD
            logger.info("Starting data download")
            data_ingestion.download_file()

            # EXTRACT ONLY IF NEEDED
            if not os.path.exists(unzip_dir) or len(os.listdir(unzip_dir)) == 0:
                logger.info("Extracting dataset")
                data_ingestion.extract_zip_file()
            else:
                logger.info("Data already extracted. Skipping extraction.")

            # SPLIT DATA (ONLY ONCE)
            train_dir = os.path.join(unzip_dir, "train")
            valid_dir = os.path.join(unzip_dir, "valid")

            if not (os.path.exists(train_dir) and os.path.exists(valid_dir)):
                logger.info("Splitting dataset into train and validation")
                data_ingestion.split_data()
            else:
                logger.info("Dataset already split. Skipping splitting.")

            logger.info("Data ingestion completed successfully")

        except Exception as e:
            logger.error("Error in Data Ingestion Pipeline")
            logger.exception(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = DataIngestionTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e