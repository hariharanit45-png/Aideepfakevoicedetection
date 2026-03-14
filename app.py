import streamlit as st
from utils import load_model, predict_audio
import tempfile
import os

st.set_page_config(page_title="AI Fake Voice Detection", page_icon="🎤", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Accuracy"])

if page == "Home":
    st.title("🎤 AI Fake Voice Detection")
    st.write("Upload an audio file to check if it's real or fake.")

    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'])

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Load model
        model = load_model('model.pkl')

        # Predict
        result = predict_audio(model, tmp_path)

        st.success(f"The audio is predicted to be: **{result}**")

        # Clean up
        os.unlink(tmp_path)

elif page == "About":
    st.title("About")
    st.write("""
    This application uses machine learning to detect whether an audio file contains a real human voice or a fake/AI-generated voice.
    
    **Features:**
    - Upload audio files in WAV, MP3, or FLAC format.
    - Real-time prediction using a trained Random Forest model.
    - Audio replay functionality.
    
    **How it works:**
    - Extracts MFCC features from the audio.
    - Feeds them into a machine learning model trained on real and fake voice samples.
    - Outputs whether the voice is real or fake.
    
    **Disclaimer:** This is a demo model with 96% accuracy. For production use, further training and validation are recommended.
    """)

elif page == "Accuracy":
    st.title("Model Accuracy")
    st.write("The model was trained on a dataset of real and fake audio samples.")
    st.metric("Accuracy", "96%")
    st.write("**Classification Report:**")
    st.code("""
              precision    recall  f1-score   support

           0       1.00      0.50      0.67         2
           1       0.96      1.00      0.98        25

    accuracy                           0.96        27
   macro avg       0.98      0.75      0.82        27
weighted avg       0.96      0.96      0.96        27
    """)