"""
VoiceShield - Streamlit Web Application
AI-Generated Voice Detection Interface

Run: streamlit run streamlit_app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np

# Ensure imports work from ml-service directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.predict import predict

MODEL_PATH = ROOT / "models" / "voice_model.pkl"
DEBUG_DEFAULT = os.getenv("VOICESHIELD_DEBUG", "0") == "1"

# Page configuration
st.set_page_config(
    page_title="VoiceShield - AI Voice Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .result-real {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .result-fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .confidence-text {
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ VoiceShield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Generated Voice Detection System</div>', unsafe_allow_html=True)

debug_mode = st.sidebar.checkbox("Enable debug mode", value=DEBUG_DEFAULT)

if debug_mode:
    st.write("### Debug Information")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"App root: {ROOT}")
    st.write(f"Model path: {MODEL_PATH}")
    st.write(f"Model exists: {MODEL_PATH.is_file()}")
    st.write(f"Python version: {sys.version}")
    st.write(f"Running on Streamlit Cloud: {bool(os.environ.get('STREAMLIT_SHARING_MODE'))}")

# Check if model exists
if not MODEL_PATH.is_file():
    st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Model Not Found</h3>
            <p>The trained model is not available. Please train the model first:</p>
            <code>python run_pipeline.py data</code>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Information section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **VoiceShield** uses machine learning to detect AI-generated voices:
    
    - 🎯 **One-Class Learning**: Trained exclusively on real human voices
    - 🔍 **Anomaly Detection**: Uses Isolation Forest algorithm
    - ✅ **REAL**: Voice matches natural human speech patterns
    - ⚠️ **FAKE**: Voice shows characteristics of AI generation
    
    **Supported formats**: WAV, MP3, FLAC, OGG, M4A, WEBM
    """)

# File uploader
st.markdown("### 📤 Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file to analyze",
    type=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
    help="Upload a voice recording to check if it's real or AI-generated"
)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()

    # Display file info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Filename:** {uploaded_file.name}")
    with col2:
        file_size_kb = len(uploaded_bytes) / 1024
        st.markdown(f"**Size:** {file_size_kb:.2f} KB")

    if debug_mode:
        st.write(f"Upload MIME type: {uploaded_file.type}")
        st.write(f"Upload size bytes: {len(uploaded_bytes)}")
    
    # Audio player
    st.markdown("### 🎵 Audio Preview")
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # Analyze button
    st.markdown("### 🔬 Analysis")
    
    if st.button("🚀 Analyze Voice", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing audio... This may take a moment..."):
            tmp_path = None
            try:
                # Save uploaded file temporarily
                suffix = Path(uploaded_file.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_bytes)
                    tmp_path = tmp_file.name

                if debug_mode:
                    st.write(f"Temporary file path: {tmp_path}")
                    st.write(f"Temporary file exists: {Path(tmp_path).is_file()}")
                
                # Run prediction
                label, confidence = predict(tmp_path, str(MODEL_PATH))

                if debug_mode:
                    st.write(f"Raw label: {label}")
                    st.write(f"Raw confidence (0-1): {confidence}")
                
                # Display results
                st.markdown("---")
                
                if label == "REAL":
                    st.markdown(f"""
                        <div class="result-box result-real">
                            Prediction: REAL
                            <div class="confidence-text">Confidence: {int(confidence * 100)}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("This voice appears to be from a real human speaker.")
                    
                else:  # FAKE
                    st.markdown(f"""
                        <div class="result-box result-fake">
                            Prediction: FAKE
                            <div class="confidence-text">Confidence: {int(confidence * 100)}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("This voice shows characteristics of AI generation or significant deviation from natural speech patterns.")
                
                # Confidence bar
                st.markdown("### 📊 Confidence Score")
                st.progress(confidence)
                st.caption(f"The model is {int(confidence * 100)}% confident in this prediction.")
                
                # Additional info
                st.markdown("""
                    <div class="info-box">
                        <strong>Note:</strong> This model was trained on real human voices only. 
                        Deviations from natural speech patterns are classified as FAKE (AI-generated).
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

else:
    st.markdown("""
        <div class="info-box">
            👆 Upload an audio file above to get started
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>VoiceShield v1.0 | Powered by Machine Learning</p>
        <p>Model: Isolation Forest | Features: Librosa Audio Analysis</p>
    </div>
""", unsafe_allow_html=True)
