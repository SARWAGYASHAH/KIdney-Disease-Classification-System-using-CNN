from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import ModelTrainer
from cnnClassifier import logger

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("Initializing Configuration Manager")

            config = ConfigurationManager()

            logger.info("Fetching Model Trainer Config")

            model_trainer_config = config.get_model_trainer_config()

            logger.info("Initializing Model Trainer")

            model_trainer = ModelTrainer(config=model_trainer_config)

            logger.info("Starting model training")

            model_trainer.train()

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error("Error in Model Training Pipeline")
            logger.exception(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = ModelTrainingPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e