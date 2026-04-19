import json
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import numpy as np
from PIL import Image

from kidney_disease_classifier import logger
from kidney_disease_classifier.config.configuration import PredictionConfig
from kidney_disease_classifier.utils.common import load_keras_model


class PredictionService:
    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self._model = None
        logger.info("PredictionService initialized with model path: %s", self.config.model_path)

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading prediction model from %s", self.config.model_path)
            self._model = load_keras_model(self.config.model_path, compile_model=False)
        return self._model

    def predict(self, image_source: str | Path | bytes | BinaryIO | BytesIO) -> dict:
        logger.info("Prediction started")
        try:
            image_array = self._preprocess_image(image_source)
            started_at = perf_counter()
            predictions = self.model.predict(image_array, verbose=0)[0]
            inference_time_ms = round((perf_counter() - started_at) * 1000, 2)
            predicted_index = int(np.argmax(predictions))
            probabilities = {
                class_name: float(predictions[index]) for index, class_name in enumerate(self.config.classes)
            }
            result = {
                "class": self.config.classes[predicted_index],
                "confidence": float(predictions[predicted_index]),
                "confidence_percent": round(float(predictions[predicted_index]) * 100, 2),
                "probabilities": probabilities,
                "inference_time_ms": inference_time_ms,
            }
            logger.info("Prediction finished successfully: %s", result)
            return result
        except Exception as error:
            if self._should_use_helper(error):
                logger.warning("Falling back to helper prediction runner due to model loading issue: %s", error)
                return self._predict_via_helper(image_source)
            logger.exception("Prediction failed: %s", error)
            raise

    def _preprocess_image(self, image_source: str | Path | bytes | BinaryIO | BytesIO) -> np.ndarray:
        image = self._open_image(image_source).convert("RGB")
        image = image.resize(tuple(self.config.image_size[:2]))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)

    def _open_image(self, image_source: str | Path | bytes | BinaryIO | BytesIO) -> Image.Image:
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source)

        if isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source))

        if hasattr(image_source, "seek"):
            image_source.seek(0)

        return Image.open(image_source)

    def _should_use_helper(self, error: Exception) -> bool:
        error_message = str(error)
        compatibility_markers = (
            "Could not deserialize class",
            "keras.src.models.functional",
            "quantization_config",
            "'str' object has no attribute 'as_list'",
        )
        return any(marker in error_message for marker in compatibility_markers)

    def _predict_via_helper(self, image_source: str | Path | bytes | BinaryIO | BytesIO) -> dict:
        helper_python = self._resolve_helper_python()
        if helper_python is None:
            raise RuntimeError("Prediction failed and no compatible helper Python interpreter was found.")

        helper_script = Path(__file__).resolve().parents[3] / "predict_helper.py"
        if not helper_script.exists():
            raise FileNotFoundError(f"Prediction helper script not found at {helper_script}")

        temp_input_path = self._materialize_image_source(image_source)
        temp_output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_output_path = Path(temp_output.name)
        temp_output.close()

        try:
            command = [
                str(helper_python),
                str(helper_script),
                str(temp_input_path),
                str(temp_output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Helper prediction failed.")

            with open(temp_output_path, "r", encoding="utf-8") as output_file:
                return json.load(output_file)
        finally:
            if temp_input_path.exists() and temp_input_path.name.startswith("renalyze_input_"):
                temp_input_path.unlink(missing_ok=True)
            temp_output_path.unlink(missing_ok=True)

    def _materialize_image_source(self, image_source: str | Path | bytes | BinaryIO | BytesIO) -> Path:
        if isinstance(image_source, (str, Path)):
            return Path(image_source)

        if isinstance(image_source, bytes):
            raw_bytes = image_source
        else:
            if hasattr(image_source, "seek"):
                image_source.seek(0)
            raw_bytes = image_source.read()

        temp_input = tempfile.NamedTemporaryFile(prefix="renalyze_input_", suffix=".jpg", delete=False)
        temp_input.write(raw_bytes)
        temp_input_path = Path(temp_input.name)
        temp_input.close()
        return temp_input_path

    def _resolve_helper_python(self) -> Path | None:
        env_override = os.getenv("RENALYZE_MODEL_PYTHON")
        if env_override:
            override_path = Path(env_override)
            if override_path.exists():
                return override_path

        current_python = Path(sys.executable)
        if "envs" in current_python.parts:
            envs_index = current_python.parts.index("envs")
            if envs_index >= 1:
                base_python = Path(*current_python.parts[:envs_index]) / "python.exe"
                if base_python.exists():
                    return base_python

        return None
