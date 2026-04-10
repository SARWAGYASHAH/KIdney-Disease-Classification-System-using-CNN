from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
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

            logger.info("Starting data download")
            data_ingestion.download_file()

            logger.info("Starting data extraction")
            data_ingestion.extract_zip_file()

            logger.info("Fixing folder structure if needed")
            data_ingestion.flatten_folder()

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