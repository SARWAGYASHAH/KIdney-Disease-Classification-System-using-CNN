import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from kidney_disease_classifier.pipeline.prediction_pipeline import PredictionPipeline


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: predict_helper.py <input_image_path> <output_json_path>")

    input_image_path = Path(sys.argv[1])
    output_json_path = Path(sys.argv[2])

    pipeline = PredictionPipeline()
    prediction = pipeline.predict(input_image_path)

    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(prediction, output_file)


if __name__ == "__main__":
    main()
