# app/symbolic.py

from typing import List


def generate_explanation(features: List[float], prediction: float) -> str:
    """
    Generate a symbolic explanation for the prediction.
    
    Args:
        features (List[float]): Input features.
        prediction (float): Neural network prediction probability.
        
    Returns:
        str: A human-readable explanation.
    """
    explanation_parts = []

    # Advanced thresholds could be data-driven; here we use fixed example thresholds
    threshold_feature_1 = 0.0
    threshold_feature_2 = 0.0

    if features[0] > threshold_feature_1:
        explanation_parts.append("Feature 1 is above baseline, contributing positively.")
    else:
        explanation_parts.append("Feature 1 is below baseline, contributing negatively.")

    if features[1] > threshold_feature_2:
        explanation_parts.append("Feature 2 is above baseline, reinforcing a positive decision.")
    else:
        explanation_parts.append("Feature 2 is below baseline, reinforcing a negative decision.")

    final_decision = "Positive" if prediction >= 0.5 else "Negative"
    explanation = (
        f"Final decision: {final_decision} (Confidence: {prediction:.2f}). "
        + " ".join(explanation_parts)
    )
    return explanation
