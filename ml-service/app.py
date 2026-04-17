"""
VoiceShield - Web API for voice REAL/FAKE prediction.

Recommended usage (from `ml-service/`):
  uvicorn app:app --host 0.0.0.0 --port 8000

This service is intended to be called by a frontend backend (e.g. Node/Express)
via multipart upload: POST /predict (field name: `file`).
"""

import os
import sys
import tempfile
from pathlib import Path

# Run from ml-service so paths and imports resolve
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.predict import predict_with_details

DEFAULT_MODEL_PATH = ROOT / "models" / "voice_model.pkl"
LEGACY_MODEL_PATH = ROOT.parent / "model.pkl"
MODEL_PATH = Path(
  os.getenv(
    "VOICESHIELD_MODEL_PATH",
    str(LEGACY_MODEL_PATH if LEGACY_MODEL_PATH.is_file() else DEFAULT_MODEL_PATH),
  )
).resolve()
MAX_FILE_SIZE_MB = int(os.getenv("VOICESHIELD_MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(title="VoiceShield", description="AI-generated voice detection")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_INDEX = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoiceShield</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 520px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #1a1a2e; margin-bottom: 0.5rem; }
    p { color: #555; margin-bottom: 1.5rem; }
    form { background: #f8f9fa; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; }
    input[type="file"] { margin: 0.5rem 0 1rem; display: block; }
    button { background: #1a1a2e; color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; font-size: 1rem; }
    button:disabled { opacity: 0.7; cursor: not-allowed; }
    button:hover:not(:disabled) { background: #16213e; }
    #result { margin-top: 1rem; display: none; }
    .result { padding: 1rem; border-radius: 8px; font-weight: 600; }
    .result.real { background: #d4edda; color: #155724; }
    .result.fake { background: #f8d7da; color: #721c24; }
    .confidence { font-size: 0.95rem; font-weight: normal; margin-top: 0.25rem; }
    .error { background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>VoiceShield</h1>
  <p>Upload an audio file to check if the voice is <strong>REAL</strong> or <strong>FAKE</strong> (AI-generated).</p>
  <form id="form">
    <label for="file">Audio file (WAV, MP3, etc.):</label>
    <input type="file" id="file" name="file" accept="audio/*" required>
    <button type="submit" id="btn">Check voice</button>
  </form>
  <div id="result"></div>
  <p style="font-size: 0.9rem; color: #666;">Model trained on real human voices only. Deviations are classified as FAKE.</p>
  <script>
    document.getElementById("form").onsubmit = async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById("file");
      const btn = document.getElementById("btn");
      const resultEl = document.getElementById("result");
      if (!fileInput.files.length) return;
      const fd = new FormData();
      fd.append("file", fileInput.files[0]);
      btn.disabled = true;
      resultEl.style.display = "block";
      resultEl.innerHTML = "<p>Analyzing...</p>";
      try {
        const r = await fetch("/predict", { method: "POST", body: fd });
        const data = await r.json();
        if (!r.ok) {
          resultEl.innerHTML = '<div class="error">' + (data.detail || data.message || "Error") + '</div>';
          return;
        }
        const cls = data.prediction === "REAL" ? "real" : "fake";
        resultEl.innerHTML = '<div class="result ' + cls + '">Prediction: ' + data.prediction + '<br><span class="confidence">Confidence: ' + data.confidence + '%</span></div>';
      } catch (err) {
        resultEl.innerHTML = '<div class="error">Request failed: ' + err.message + '</div>';
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the upload form."""
    return HTML_INDEX


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an audio file, run VoiceShield, return prediction and confidence.
    """
    if not file.filename or not file.filename.lower().strip():
        raise HTTPException(status_code=400, detail="No file selected")

    suffix = Path(file.filename).suffix or ".wav"
    if suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}:
        suffix = ".wav"

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB} MB",
        )

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train first: python run_pipeline.py data",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_with_details(tmp_path, str(MODEL_PATH))
        label = result["label"]
        confidence = result["confidence"]
        return {
            "success": True,
            "prediction": label,
            "confidence": round(confidence * 100),
            "confidence_0_to_1": confidence,
            "model_type": result.get("model_type"),
            "raw_score": result.get("raw_score"),
            "file_name": file.filename,
            "file_size_bytes": len(contents),
            "model_loaded": True,
            "message": f"Prediction: {label}, Confidence: {round(confidence * 100)}%",
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/health")
def health():
    """Health check; reports if model is loaded."""
    model_type = None
    model_error = None

    if MODEL_PATH.is_file():
        try:
            model_type = "legacy_binary_classifier" if MODEL_PATH.name.lower() == "model.pkl" else "one_class_iforest"
        except Exception as e:
            model_error = str(e)

    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "model_type": model_type,
        "model_error": model_error,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
    }
