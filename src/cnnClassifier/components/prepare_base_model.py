import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger  # ✅ ADD THIS


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        '''Stores configuration'''
        self.config = config

    def get_base_model(self):
        logger.info("🔹 Loading VGG16 base model")

        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        logger.info("🔹 Saving base model")
        self.save_model(path=self.config.base_model_path, model=self.model)

        logger.info(f"✅ Base model saved at: {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        logger.info("🔹 Preparing full model (Freezing + Adding layers)")

        '''Freeze layers'''
        if freeze_all:
            logger.info("🔹 Freezing all layers")
            for layer in model.layers:
                layer.trainable = False

        elif (freeze_till is not None) and (freeze_till > 0):
            logger.info(f"🔹 Freezing layers till: {freeze_till}")
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        '''Add classification head'''
        logger.info("🔹 Adding custom classification head")

        flatten_in = tf.keras.layers.Flatten()(model.output)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        '''Compile model'''
        logger.info("🔹 Compiling model")

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()

        logger.info("✅ Full model prepared successfully")

        return full_model

    def update_base_model(self):
        logger.info("🔹 Updating base model")

        if not hasattr(self, "model"):
            logger.error("❌ Base model not found. Run get_base_model() first.")
            raise ValueError("Base model not found. Run get_base_model() first.")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        logger.info("🔹 Saving updated base model")
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

        logger.info(f"✅ Updated model saved at: {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        logger.info(f"🔹 Saving model at: {path}")
        model.save(path)