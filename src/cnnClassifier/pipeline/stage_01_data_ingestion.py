from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger
import os

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    '''this class makes directories required for data ingestion and downloads the data and then unzips it'''

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

            # ✅ PATHS
            zip_path = data_ingestion_config.local_data_file
            unzip_dir = os.path.dirname(zip_path)

            # ✅ DOWNLOAD (safe: already has check inside)
            logger.info("Starting data download")
            data_ingestion.download_file()

            # ✅ EXTRACT ONLY IF NOT ALREADY DONE
            if not os.path.exists(unzip_dir) or len(os.listdir(unzip_dir)) == 0:
                logger.info("Starting data extraction")
                data_ingestion.extract_zip_file()
            else:
                logger.info("Data already extracted. Skipping extraction.")

            # ✅ FLATTEN ONLY IF NEEDED
            expected_folder = os.path.join(unzip_dir, "train")  # 🔁 change if needed

            if not os.path.exists(expected_folder):
                logger.info("Fixing folder structure")
                data_ingestion.flatten_folder()
            else:
                logger.info("Folder structure already correct. Skipping flattening.")

            logger.info("Data ingestion completed successfully")

        except Exception as e:
            logger.error("Error in Data Ingestion Pipeline")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = DataIngestionTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e