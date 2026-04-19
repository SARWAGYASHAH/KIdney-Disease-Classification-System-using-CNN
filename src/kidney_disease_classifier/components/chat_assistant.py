from __future__ import annotations

from kidney_disease_classifier import logger


CLASS_EXPLANATIONS = {
    "Normal": (
        "The scan pattern is most consistent with a normal kidney appearance. "
        "That means the model did not find strong signs of cyst, stone, or tumor in this image."
    ),
    "Tumor": (
        "The model sees a pattern more consistent with a renal tumor class. "
        "This is a screening-style AI result, not a diagnosis, so specialist review and follow-up imaging are important."
    ),
    "Cyst": (
        "The model sees a pattern more consistent with a renal cyst class. "
        "Many cysts are benign, but imaging review is still needed to understand the type and significance."
    ),
    "Stone": (
        "The model sees a pattern more consistent with a kidney stone class. "
        "Stones can be associated with obstruction or pain, so symptoms and imaging review matter."
    ),
}


NEXT_STEPS = {
    "Normal": "Keep this as a reassuring result, but confirm with the treating clinician if symptoms are still present.",
    "Tumor": "Share the scan with a radiologist or urologist and consider contrast imaging or further workup.",
    "Cyst": "Ask for radiology review to determine whether the cyst appears simple or needs closer follow-up.",
    "Stone": "Review symptoms, hydration status, and whether additional imaging or pain management is needed.",
}


class ChatAssistant:
    def respond(self, question: str, prediction: dict | None = None) -> str:
        question_normalized = (question or "").strip().lower()
        logger.info("Chat assistant received question: %s", question_normalized)

        if not question_normalized:
            return "Ask me about the prediction, confidence, class meaning, or suggested next steps."

        if any(token in question_normalized for token in ["hello", "hi", "hey"]):
            return "Hello. I can explain the prediction, confidence level, class probabilities, and what each class means."

        if prediction and any(token in question_normalized for token in ["result", "prediction", "current", "show"]):
            return self._summarize_prediction(prediction)

        if prediction and any(token in question_normalized for token in ["confidence", "certain", "reliable", "sure"]):
            return self._confidence_response(prediction)

        if prediction and any(token in question_normalized for token in ["next step", "what should", "what now", "treatment", "doctor"]):
            predicted_class = prediction.get("class", "Normal")
            return (
                f"Based on the current AI result, the predicted class is {predicted_class}. "
                f"{NEXT_STEPS.get(predicted_class, 'Clinical review is recommended.')}"
            )

        for class_name in CLASS_EXPLANATIONS:
            if class_name.lower() in question_normalized:
                return CLASS_EXPLANATIONS[class_name]

        if "probab" in question_normalized and prediction:
            return self._probability_breakdown(prediction)

        if any(token in question_normalized for token in ["model", "architecture", "cnn"]):
            return (
                "Renalyze is using your custom CNN kidney disease classifier with four output classes: "
                "Cyst, Normal, Stone, and Tumor."
            )

        return (
            "I can help explain the current prediction, confidence, class probabilities, or what Normal, Cyst, Stone, "
            "and Tumor mean in the app."
        )

    def _summarize_prediction(self, prediction: dict) -> str:
        predicted_class = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence_percent", 0.0)
        return (
            f"The current prediction is {predicted_class} with {confidence:.2f}% confidence. "
            f"{CLASS_EXPLANATIONS.get(predicted_class, '')}"
        ).strip()

    def _confidence_response(self, prediction: dict) -> str:
        confidence = float(prediction.get("confidence_percent", 0.0))
        predicted_class = prediction.get("class", "Unknown")
        if confidence >= 95:
            confidence_band = "very high"
        elif confidence >= 80:
            confidence_band = "strong"
        elif confidence >= 60:
            confidence_band = "moderate"
        else:
            confidence_band = "limited"

        return (
            f"The model confidence for {predicted_class} is {confidence:.2f}%, which is a {confidence_band} signal. "
            "Confidence reflects model certainty on this image, not a final medical diagnosis."
        )

    def _probability_breakdown(self, prediction: dict) -> str:
        probabilities = prediction.get("probabilities", {})
        if not probabilities:
            return "No probability breakdown is available yet."

        ordered = ", ".join(
            f"{class_name} {float(score) * 100:.2f}%"
            for class_name, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        )
        return f"Class probability breakdown: {ordered}."
