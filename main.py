import streamlit as st
from transformers import pipeline
import tempfile
import os
import re
import logging
import pathlib
from audio_recorder_streamlit import audio_recorder

# Configuration - Remove problematic HTTP backend configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # Increased timeout

# File paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_MAPPING = {
    "yes": VIDEO_DIR / "yes.mp4",
    "no": VIDEO_DIR / "no.mp4"
}

# Verify files
for word, path in VIDEO_MAPPING.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing video: {path.name}")

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load model with PyTorch compatibility fix"""
    try:
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",  # Smaller model for testing
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        st.stop()

def process_audio(audio_bytes):
    """Handle audio processing without temp files"""
    try:
        pipe = load_whisper_model()
        result = pipe({
            "raw": audio_bytes,
            "sampling_rate": 16000
        })
        return clean_text(result["text"])
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return None

def clean_text(text):
    """Text normalization"""
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def main():
    # Disable problematic watcher
    st.config.set_option("server.enableCORS", False)
    st.config.set_option("server.enableXsrfProtection", False)
    
    st.title("🎤 Voice-Activated Video Player")
    
    # Audio input
    audio_bytes = audio_recorder()
    if audio_bytes:
        with st.spinner("Analyzing..."):
            text = process_audio(audio_bytes)
            if text:
                show_results(text)

def show_results(text):
    st.subheader("Results")
    st.write(f"Recognized: {text}")
    
    if text in VIDEO_MAPPING:
        st.video(str(VIDEO_MAPPING[text]))
    else:
        st.warning("No matching video")

if __name__ == "__main__":
    main()
