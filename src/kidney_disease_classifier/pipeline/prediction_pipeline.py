from kidney_disease_classifier import logger
from kidney_disease_classifier.components.prediction import PredictionService
from kidney_disease_classifier.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self, configuration_manager: ConfigurationManager | None = None) -> None:
        self.configuration_manager = configuration_manager or ConfigurationManager()
        self.prediction_service = PredictionService(self.configuration_manager.get_prediction_config())

    def predict(self, image_source):
        logger.info("Prediction pipeline invoked")
        return self.prediction_service.predict(image_source)
