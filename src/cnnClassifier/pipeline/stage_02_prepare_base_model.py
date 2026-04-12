from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("🔹 Initializing Configuration Manager")
            config = ConfigurationManager()

            logger.info("🔹 Fetching Prepare Base Model Config")
            prepare_base_model_config = config.get_prepare_base_model_config()

            logger.info("🔹 Initializing PrepareBaseModel Component")
            prepare_base_model = PrepareBaseModel(
                config=prepare_base_model_config
            )

            logger.info("🔹 Loading VGG16 Base Model")
            prepare_base_model.get_base_model()

            logger.info("🔹 Updating Base Model (Adding Custom Layers)")
            prepare_base_model.update_base_model()

            logger.info("✅ Prepare Base Model Stage Completed Successfully")

        except Exception as e:
            logger.error("❌ Error occurred in Prepare Base Model Stage")
            logger.exception(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = PrepareBaseModelTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")

    except Exception as e:
        logger.exception(e)
        raise e