from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Prepare Base Model Stage"

if __name__ == "__main__":
    try:
        logger.info("\n" + "=" * 20 + f" {STAGE_NAME} STARTED " + "=" * 20)

        prepare_base_model = PrepareBaseModelTrainingPipeline()
        prepare_base_model.main()

        logger.info("=" * 20 + f" {STAGE_NAME} COMPLETED " + "=" * 20 + "\n")

    except Exception as e:
        logger.error(f"❌ Error in {STAGE_NAME}")
        logger.exception(e)
        raise e