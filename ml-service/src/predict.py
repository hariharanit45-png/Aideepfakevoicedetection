"""
VoiceShield - Prediction module.
Preprocess → extract features → load model → classify REAL or FAKE with confidence.
"""

import pickle
import numpy as np
import sys
from pathlib import Path
import librosa

import joblib

from .preprocess import load_and_preprocess
from .extract_features import extract_features_from_waveform, TARGET_SR

MODEL_SAVE_PATH = "models/voice_model.pkl"


def _extract_legacy_binary_features(audio_path: str) -> np.ndarray:
    """
    Extract features compatible with the legacy RandomForest `model.pkl`.
    This matches the root-level `utils.extract_features` pipeline.
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    pitch = 0.0
    jitter = 0.0
    energy = float(np.mean(librosa.feature.rms(y=y)))
    features = np.concatenate([mfccs_mean, [pitch, jitter, energy]])
    return features.astype(np.float64)


def _setup_numpy_compatibility():
    """Set up compatibility layer for numpy._core module reference"""
    import numpy
    # Handle both directions: numpy 1.x → 2.x and 2.x → 1.x
    if not hasattr(numpy, '_core') and hasattr(numpy, 'core'):
        # NumPy < 2.0: add _core as alias to core
        sys.modules['numpy._core'] = numpy.core
    elif hasattr(numpy, '_core') and not hasattr(numpy, 'core'):
        # NumPy >= 2.0: add core as alias to _core
        sys.modules['numpy.core'] = numpy._core


def _load_artifact(model_path: str):
    """Load model artifact with compatibility handling (pickle/joblib)."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Setup compatibility layer BEFORE loading pickle
    _setup_numpy_compatibility()
    
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if 'numpy' in str(e):
            raise ModuleNotFoundError(
                f"NumPy compatibility issue when loading model: {e}\n"
                "This usually occurs when the model was saved with a different NumPy version.\n"
                "Try reinstalling NumPy: pip install --upgrade numpy>=2.0.0"
            ) from e
        raise
    except Exception as e:
        try:
            # Fallback for artifacts saved via joblib.dump (e.g. legacy model.pkl)
            return joblib.load(path)
        except Exception:
            print(f"Error loading model: {e}")
            raise


def _predict_one_class(artifact: dict, audio_path: str) -> tuple[str, float, float]:
    """Predict using one-class artifact dict: {model, scaler, feature_columns}."""
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    waveform = load_and_preprocess(audio_path)
    feature_vector = extract_features_from_waveform(waveform, TARGET_SR)
    X = np.array([feature_vector])
    if X.shape[1] != len(feature_columns):
        raise ValueError(
            f"Feature dimension mismatch: got {X.shape[1]}, expected {len(feature_columns)}. "
            "Retrain the model with the current extract_features pipeline."
        )
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]  # 1 = inlier (REAL), -1 = outlier (FAKE)
    decision = float(model.decision_function(X_scaled)[0])
    confidence = _decision_to_confidence(decision, model)
    label = "REAL" if prediction == 1 else "FAKE"
    return label, confidence, decision


def _predict_legacy_binary(model, audio_path: str) -> tuple[str, float, float]:
    """Predict using legacy sklearn binary classifier (0=FAKE, 1=REAL)."""
    features = _extract_legacy_binary_features(audio_path)
    X = np.array([features])
    pred = int(model.predict(X)[0])
    prob_real = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            class_to_prob = {int(c): float(p) for c, p in zip(model.classes_, proba)}
            prob_real = class_to_prob.get(1)
        except Exception:
            prob_real = None

    if pred == 1:
        label = "REAL"
        confidence = prob_real if prob_real is not None else 0.5
    else:
        label = "FAKE"
        confidence = (1.0 - prob_real) if prob_real is not None else 0.5
    raw_score = prob_real if prob_real is not None else float(pred)
    return label, float(np.clip(confidence, 0, 1)), float(raw_score)


def _artifact_model_type(artifact) -> str:
    """Return model type identifier for diagnostics."""
    if isinstance(artifact, dict) and {"model", "scaler", "feature_columns"}.issubset(artifact.keys()):
        return "one_class_iforest"
    if hasattr(artifact, "predict"):
        return "legacy_binary_classifier"
    raise ValueError("Unsupported model artifact format")


def predict_with_details(audio_path: str, model_path: str = MODEL_SAVE_PATH) -> dict:
    """Run prediction and return label/confidence plus model diagnostics."""
    artifact = _load_artifact(model_path)
    model_type = _artifact_model_type(artifact)

    if model_type == "one_class_iforest":
        label, confidence, raw_score = _predict_one_class(artifact, audio_path)
    else:
        label, confidence, raw_score = _predict_legacy_binary(artifact, audio_path)

    return {
        "label": label,
        "confidence": float(confidence),
        "model_type": model_type,
        "raw_score": float(raw_score),
    }


def predict(audio_path: str, model_path: str = MODEL_SAVE_PATH) -> tuple[str, float]:
    """
    Run VoiceShield on a single audio file.

    Args:
        audio_path: Path to the audio file.
        model_path: Path to the saved model pickle.

    Returns:
        (label, confidence): "REAL" or "FAKE", and confidence in [0, 1].
    """
    result = predict_with_details(audio_path, model_path)
    return result["label"], result["confidence"]


def _decision_to_confidence(decision: float, model) -> float:
    """
    Map Isolation Forest decision_function to a 0–1 confidence score.
    Higher decision = more inlier-like = higher confidence for REAL.
    """
    # decision_function: negative = more anomalous. We want REAL → high confidence.
    # Simple sigmoid-like scaling: clip and normalize to [0, 1]
    try:
        # Typical range is roughly [-0.5, 0.5] for sklearn's Isolation Forest
        shifted = decision + 0.5
        confidence = np.clip(shifted, 0, 1)
        return float(confidence)
    except Exception:
        return 0.5


def predict_and_print(audio_path: str, model_path: str = MODEL_SAVE_PATH) -> None:
    """
    Run prediction and print result in the required format.
    Example output:
      Prediction: REAL
      Confidence: 91%
    """
    label, confidence = predict(audio_path, model_path)
    pct = int(round(confidence * 100))
    print(f"Prediction: {label}")
    print(f"Confidence: {pct}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_path> [model_path]")
        sys.exit(1)
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_SAVE_PATH
    predict_and_print(audio_path, model_path)
