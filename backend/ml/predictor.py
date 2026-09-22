"""
Runtime loader for the trained mastery-gain neural network
(train_gain_predictor.py). Loaded once at server startup, held in memory,
never calls out to any external API or network -- this is a real, local,
pre-trained model file (gain_predictor.joblib), not a wrapper around a
hosted LLM.

If the model file is missing (e.g. a fresh clone before anyone has run
the training script), this fails LOUDLY rather than silently
substituting a fake prediction -- see main.py's startup check.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).parent / "gain_predictor.joblib"

_model = None


def is_available() -> bool:
    return MODEL_PATH.exists()


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python backend/ml/train_gain_predictor.py` "
                "once to train and save the model before starting the server."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_gain(
    mastery_before: float,
    resource_difficulty: float,
    duration_ratio: float,
    prereq_satisfied: bool,
    resource_quality: float,
) -> float:
    """
    Real neural network forward pass (scikit-learn MLPRegressor.predict,
    which runs the actual learned weight matrices -- matrix multiply +
    ReLU per layer -- entirely in-process). Returns a predicted mastery
    gain in roughly [0, 1] (not hard-clamped; see train_gain_predictor.py
    for how the label was generated).

    Feature order MUST match train_gain_predictor.py exactly.
    """
    model = load_model()
    duration_ratio = min(duration_ratio, 2.0)
    x = np.array([[mastery_before, resource_difficulty, duration_ratio,
                    1.0 if prereq_satisfied else 0.0, resource_quality]])
    return float(model.predict(x)[0])
