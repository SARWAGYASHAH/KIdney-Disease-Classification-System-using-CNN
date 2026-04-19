import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from kidney_disease_classifier import logger
from kidney_disease_classifier.components.data_ingestion import DataIngestion
from kidney_disease_classifier.config.configuration import ConfigurationManager
from kidney_disease_classifier.pipeline.evaluation_pipeline import EvaluationPipeline


def run_pipeline() -> None:
    logger.info("Pipeline execution started")
    configuration = ConfigurationManager()

    try:
        logger.info("Stage 1: Data Ingestion started")
        data_ingestion = DataIngestion(configuration.get_data_ingestion_config())
        data_ingestion.run()
        logger.info("Stage 1: Data Ingestion completed")

        logger.info("Stage 2: Model Evaluation started")
        evaluation_pipeline = EvaluationPipeline(configuration)
        evaluation_pipeline.run()
        logger.info("Stage 2: Model Evaluation completed")
    except Exception as error:
        logger.exception("Pipeline execution failed: %s", error)
        raise

    logger.info("Pipeline execution finished successfully")


if __name__ == "__main__":
    run_pipeline()
