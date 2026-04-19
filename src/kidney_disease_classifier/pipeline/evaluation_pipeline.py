from kidney_disease_classifier import logger
from kidney_disease_classifier.components.model_evaluation import ModelEvaluation
from kidney_disease_classifier.config.configuration import ConfigurationManager


class EvaluationPipeline:
    def __init__(self, configuration_manager: ConfigurationManager | None = None) -> None:
        self.configuration_manager = configuration_manager or ConfigurationManager()

    def run(self) -> dict:
        logger.info("Evaluation pipeline started")
        try:
            evaluation = ModelEvaluation(self.configuration_manager.get_model_evaluation_config())
            result = evaluation.run()
            logger.info("Evaluation pipeline completed successfully")
            return result
        except Exception as error:
            logger.exception("Evaluation pipeline failed: %s", error)
            raise
