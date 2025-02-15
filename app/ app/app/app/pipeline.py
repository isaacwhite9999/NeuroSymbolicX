# app/pipeline.py

import argparse
import asyncio
import logging
from typing import Dict, List

import numpy as np
import torch

from app.model import load_model
from app.symbolic import generate_explanation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def predict_with_explanation(features: List[float]) -> Dict[str, object]:
    """
    Given a list of features, predict the outcome and provide a symbolic explanation.
    
    Args:
        features (List[float]): List of numerical features.
        
    Returns:
        Dict[str, object]: Dictionary with keys 'prediction' and 'explanation'.
    """
    try:
        model = load_model()
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise

    input_array = np.array(features, dtype=np.float32).reshape(1, -1)
    input_tensor = torch.tensor(input_array)
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except Exception as e:
        logger.exception("Error during model inference: %s", e)
        raise

    prediction = output.item()
    explanation = generate_explanation(features, prediction)
    return {"prediction": prediction, "explanation": explanation}


async def main_async(args: argparse.Namespace) -> None:
    """
    Asynchronous main entrypoint.
    """
    result = predict_with_explanation(args.features)
    print("Prediction:", result["prediction"])
    print("Explanation:", result["explanation"])


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroSymbolicX: Advanced Neuro-Symbolic AI Pipeline")
    parser.add_argument(
        "--features",
        nargs=2,
        type=float,
        required=True,
        help="Two numerical features (e.g., --features 0.5 -0.3)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
