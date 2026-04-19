import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from kidney_disease_classifier import logger


class ConfigNode(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def read_yaml(path_to_yaml: Path) -> ConfigNode:
    logger.info("Reading YAML file from %s", path_to_yaml)
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file) or {}
    return _to_config_node(content)


def create_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", path)


def save_json(path: Path, data: dict) -> None:
    logger.info("Saving JSON file to %s", path)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def load_keras_model(model_path: str | Path, compile_model: bool = False) -> tf.keras.Model:
    resolved_model_path = Path(model_path)

    try:
        logger.info("Loading Keras model from %s", resolved_model_path)
        return tf.keras.models.load_model(resolved_model_path, compile=compile_model)
    except Exception as error:
        error_message = str(error)
        compatibility_markers = (
            "quantization_config",
            "keras.src.models.functional",
            "Could not deserialize class 'Functional'",
        )

        if not any(marker in error_message for marker in compatibility_markers):
            raise

        logger.warning(
            "Standard model loading failed due to Keras config compatibility. "
            "Retrying with a sanitized temporary model archive."
        )
        sanitized_model_path = _create_sanitized_keras_archive(resolved_model_path)
        return tf.keras.models.load_model(sanitized_model_path, compile=compile_model)


def _create_sanitized_keras_archive(model_path: Path) -> str:
    with zipfile.ZipFile(model_path, "r") as source_zip:
        config_data = json.loads(source_zip.read("config.json"))
        sanitized_config = _remove_problematic_keras_keys(config_data)

        temp_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        with zipfile.ZipFile(temp_path, "w") as target_zip:
            for member in source_zip.infolist():
                if member.filename == "config.json":
                    target_zip.writestr(member, json.dumps(sanitized_config))
                else:
                    target_zip.writestr(member, source_zip.read(member.filename))

    logger.info("Created sanitized temporary model archive at %s", temp_path)
    return str(temp_path)


def _remove_problematic_keras_keys(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            return value.get("config", {}).get("name", "float32")

        sanitized = {}
        for key, item in value.items():
            if key == "quantization_config":
                continue
            if key == "optional":
                continue

            rewritten_key = "batch_input_shape" if key == "batch_shape" else key
            rewritten_item = _remove_problematic_keras_keys(item)
            sanitized[rewritten_key] = _rewrite_keras_module_path(rewritten_item) if rewritten_key == "module" else rewritten_item
        return sanitized
    if isinstance(value, list):
        return [_remove_problematic_keras_keys(item) for item in value]
    return value


def _rewrite_keras_module_path(module_value: Any) -> Any:
    tensorflow_version = getattr(tf, "__version__", "")
    if tensorflow_version.startswith("2.15"):
        module_mappings = {
            "keras.src.models.functional": "keras.src.engine.functional",
            "keras.src.models.sequential": "keras.src.engine.sequential",
        }
    else:
        module_mappings = {
            "keras.src.engine.functional": "keras.src.models.functional",
            "keras.src.engine.sequential": "keras.src.models.sequential",
        }
    return module_mappings.get(module_value, module_value)
