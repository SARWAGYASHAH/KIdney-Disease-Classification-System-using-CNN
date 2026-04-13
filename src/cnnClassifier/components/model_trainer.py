import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        try:
            logger.info("Loading updated base model")

            model = tf.keras.models.load_model(
                self.config.updated_base_model_path
            )

            logger.info("Model loaded successfully")

            # Data Generators
            logger.info("Initializing data generators")

            if self.config.params_augmentation:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True
                )
            else:
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255
                )

            valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
            )

            # Training Data
            logger.info("Loading training data")

            train_generator = train_datagen.flow_from_directory(
                self.config.training_data,
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode="categorical"
            )

            # Validation Data
            logger.info("Loading validation data")

            valid_generator = valid_datagen.flow_from_directory(
                self.config.validation_data,
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                class_mode="categorical"
            )

            # Training
            logger.info("Starting model training")

            model.fit(
                train_generator,
                epochs=self.config.params_epochs,
                validation_data=valid_generator
            )

            logger.info("Model training completed")

            # Save model
            logger.info("Saving trained model")

            model.save(self.config.trained_model_path)

            logger.info(f"Model saved at {self.config.trained_model_path}")

        except Exception as e:
            logger.error("Error during model training")
            logger.exception(e)
            raise e