# Kidney Disease Classification

Production-ready Kidney Disease Classification project using a custom trained CNN model for four classes:

- Cyst
- Normal
- Stone
- Tumor

The repository is structured for inference, evaluation, and deployment. It intentionally excludes any pretrained-model preparation stage or local training pipeline because the model has already been trained externally in Colab.

## Project Structure

```text
KidneyDiseaseClassification/
├── artifacts/
│   ├── data_ingestion/
│   │   ├── raw/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── model/
│   │   └── best_kidney_model.keras
│   └── model_evaluation/
│       ├── confusion_matrix.png
│       └── scores.json
├── config/
│   └── config.yaml
├── logs/
│   └── running_logs.log
├── src/
│   └── kidney_disease_classifier/
│       ├── __init__.py
│       ├── logger.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── model_evaluation.py
│       │   └── prediction.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── configuration.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── evaluation_pipeline.py
│       │   └── prediction_pipeline.py
│       └── utils/
│           ├── __init__.py
│           └── common.py
├── app.py
├── Dockerfile
├── main.py
├── params.yaml
├── requirements.txt
├── setup.py
└── .gitignore
```

## Features

- Data ingestion from a local dataset path or a mounted Google Drive path
- Automatic 70/15/15 train/validation/test split
- Model evaluation against the held-out test set
- JSON metrics report with accuracy, loss, and per-class classification report
- Confusion matrix generation
- Single-image prediction pipeline
- Flask API for deployment
- Centralized logging to `logs/running_logs.log`

## Expected Dataset Layout

The source dataset directory should contain one folder per class:

```text
dataset_root/
├── Cyst/
├── Normal/
├── Stone/
└── Tumor/
```

If your dataset is nested inside another folder, the ingestion stage will try to locate the class root automatically.

## Configuration

Update `config/config.yaml` with the path to your source dataset before running ingestion:

```yaml
source_data_path: "C:/path/to/your/dataset"
google_drive_data_path: null
```

For a Google Drive dataset mounted in Colab or available locally, provide that mounted folder path in `google_drive_data_path`.

## Installation

Use Python 3.10 or 3.11 for this project. TensorFlow `2.15.0` will not install on Python 3.13.

```bash
python --version
pip install -r requirements.txt
pip install -e .
```

If your current interpreter is Python 3.13, create a dedicated environment first.

### Conda

```bash
conda create -n kidney-cnn python=3.10 -y
conda activate kidney-cnn
pip install -r requirements.txt
pip install -e .
```

### venv

```bash
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run the Pipeline

```bash
python main.py
```

This runs:

1. Data ingestion
2. Model evaluation

## Run the API

```bash
python app.py
```

### Endpoints

- `GET /` returns a health status response
- `POST /predict` accepts an image file and returns predicted class and confidence

Example response:

```json
{
  "class": "Tumor",
  "confidence": 0.9963
}
```

## Docker

```bash
docker build -t kidney-disease-classifier .
docker run -p 8080:8080 kidney-disease-classifier
```

## Notes

- The trained model is expected at `artifacts/model/best_kidney_model.keras`
- Prediction preprocessing uses resize to `(224, 224)` and rescaling by `1.0 / 255.0`
- Logs are written to `logs/running_logs.log`
