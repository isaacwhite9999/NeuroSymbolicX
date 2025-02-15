# tests/test_pipeline.py

from typing import Dict

import pytest
from app.pipeline import predict_with_explanation


def test_pipeline_output() -> None:
    """
    Test that the pipeline returns a valid prediction and explanation.
    """
    features = [0.5, -0.3]
    result: Dict[str, object] = predict_with_explanation(features)
    assert "prediction" in result, "Result must contain a 'prediction' key."
    assert "explanation" in result, "Result must contain an 'explanation' key."
    assert isinstance(result["explanation"], str), "Explanation must be a string."
