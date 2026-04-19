import os
import sys
from http import HTTPStatus
from io import BytesIO
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flask import Flask, jsonify, render_template, request

from kidney_disease_classifier import logger
from kidney_disease_classifier.components.chat_assistant import ChatAssistant
from kidney_disease_classifier.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__)
prediction_pipeline = PredictionPipeline()
chat_assistant = ChatAssistant()


def load_model_metrics() -> dict:
    scores_path = PROJECT_ROOT / "artifacts" / "model_evaluation" / "scores.json"
    if not scores_path.exists():
        return {
            "test_accuracy": "N/A",
            "val_accuracy": "99.63%",
            "classes": 4,
            "image_size": "224 x 224 RGB",
        }

    try:
        with open(scores_path, "r", encoding="utf-8") as scores_file:
            scores = json.load(scores_file)
        accuracy = float(scores.get("accuracy", 0.0)) * 100
        return {
            "test_accuracy": f"{accuracy:.2f}%",
            "val_accuracy": "99.63%",
            "classes": len(scores.get("class_indices", {})) or 4,
            "image_size": "224 x 224 RGB",
        }
    except Exception as error:
        logger.exception("Failed to load model metrics for UI: %s", error)
        return {
            "test_accuracy": "N/A",
            "val_accuracy": "99.63%",
            "classes": 4,
            "image_size": "224 x 224 RGB",
        }


@app.get("/")
def index():
    logger.info("Renalyze web app accessed")
    return render_template("index.html", metrics=load_model_metrics())


@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "ok", "message": "Renalyze service is running."}), HTTPStatus.OK


@app.post("/predict")
def predict():
    logger.info("Prediction endpoint accessed")

    if "image" not in request.files:
        logger.error("Prediction failed: image file missing from request")
        return jsonify({"error": "No image file provided under 'image'."}), HTTPStatus.BAD_REQUEST

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        logger.error("Prediction failed: empty image file")
        return jsonify({"error": "Uploaded image file is empty."}), HTTPStatus.BAD_REQUEST

    try:
        image_bytes = BytesIO(image_file.read())
        prediction = prediction_pipeline.predict(image_bytes)
        logger.info("Prediction completed successfully: %s", prediction)
        return jsonify(prediction), HTTPStatus.OK
    except Exception as error:
        logger.exception("Prediction request failed: %s", error)
        return jsonify({"error": str(error)}), HTTPStatus.INTERNAL_SERVER_ERROR


@app.post("/chat")
def chat():
    logger.info("Chat endpoint accessed")
    payload = request.get_json(silent=True) or {}
    question = payload.get("message", "")
    prediction = payload.get("prediction")

    if not question:
        return jsonify({"error": "A message is required."}), HTTPStatus.BAD_REQUEST

    try:
        response = chat_assistant.respond(question, prediction)
        return jsonify({"reply": response}), HTTPStatus.OK
    except Exception as error:
        logger.exception("Chat request failed: %s", error)
        return jsonify({"error": str(error)}), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
