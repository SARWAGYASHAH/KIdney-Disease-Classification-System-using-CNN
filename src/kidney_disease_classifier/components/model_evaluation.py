import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from kidney_disease_classifier import logger
from kidney_disease_classifier.config.configuration import ModelEvaluationConfig
from kidney_disease_classifier.utils.common import create_directories, load_keras_model, save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        logger.info("ModelEvaluation initialized with model path: %s", self.config.model_path)

    def run(self) -> dict:
        logger.info("Model evaluation stage started")
        logger.info(
            "Using paths - model: %s, test data: %s, scores file: %s, confusion matrix: %s",
            self.config.model_path,
            self.config.test_data_path,
            self.config.scores_file,
            self.config.confusion_matrix_path,
        )

        try:
            model = load_keras_model(self.config.model_path, compile_model=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            test_generator = self._create_test_generator()

            evaluation_result = model.evaluate(test_generator, verbose=0)
            predictions = model.predict(test_generator, verbose=0)

            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = test_generator.classes
            ordered_class_names = [
                class_name for class_name, _ in sorted(test_generator.class_indices.items(), key=lambda item: item[1])
            ]

            report = classification_report(
                true_labels,
                predicted_labels,
                target_names=ordered_class_names,
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(true_labels, predicted_labels)

            scores = {
                "loss": float(evaluation_result[0]),
                "accuracy": float(evaluation_result[1]) if len(evaluation_result) > 1 else None,
                "class_indices": test_generator.class_indices,
                "classification_report": report,
            }

            create_directories([self.config.evaluation_dir])
            save_json(self.config.scores_file, scores)
            self._save_confusion_matrix(matrix, ordered_class_names)

            logger.info("Model evaluation stage completed successfully")
            return scores
        except Exception as error:
            logger.exception("Model evaluation stage failed: %s", error)
            raise

    def _create_test_generator(self):
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
        return data_generator.flow_from_directory(
            directory=str(self.config.test_data_path),
            target_size=tuple(self.config.image_size[:2]),
            batch_size=self.config.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    def _save_confusion_matrix(self, matrix: np.ndarray, class_names: list[str]) -> None:
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        figure.colorbar(image, ax=axis)

        axis.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = matrix.max() / 2.0 if matrix.size else 0.0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    format(matrix[row_index, column_index], "d"),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )

        figure.tight_layout()
        figure.savefig(self.config.confusion_matrix_path, bbox_inches="tight")
        plt.close(figure)
