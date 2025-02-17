# main.py
import streamlit as st
from transformers import pipeline
import tempfile
import os
import re
import logging
import pathlib
from audio_recorder_streamlit import audio_recorder
from huggingface_hub import configure_http_backend

# Configure logging and HTTP settings
configure_http_backend(backend="httpcore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

# Configuration
BASE_DIR = pathlib.Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_MAPPING = {
    "hello": VIDEO_DIR / "hello.mp4",
    "yes": VIDEO_DIR / "yes.mp4",
    "no": VIDEO_DIR / "no.mp4"
}

# Verify video files exist
for word, path in VIDEO_MAPPING.items():
    if not path.exists():
        logger.error(f"Missing video file: {str(path)}")
        raise FileNotFoundError(f"Video file missing: {path.name}")

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load Whisper model with error handling"""
    try:
        logger.info("Loading Whisper model...")
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("Failed to load speech recognition model")
        st.stop()

def process_audio(audio_bytes):
    """Process audio bytes through Whisper model"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        pipe = load_whisper_model()
        result = pipe(tmp_path)
        return clean_text(result["text"])
    
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        st.error("Error processing audio")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                logger.warning("Temporary file cleanup delayed")

def clean_text(text):
    """Normalize recognized text"""
    text = re.sub(r'[^\w\s]', '', text).strip().lower()
    return text.split()[0] if text else ""

def debug_page():
    """Hidden debug tools"""
    st.header("🔧 Debug Tools")
    
    st.subheader("File Verification")
    for word, path in VIDEO_MAPPING.items():
        st.write(f"{word}: {'✅ Exists' if path.exists() else '❌ Missing'}")
    
    st.subheader("Model Test")
    if st.button("Test Model Loading"):
        try:
            load_whisper_model()
            st.success("Model loaded successfully")
        except Exception as e:
            st.error(f"Load failed: {str(e)}")

def main():
    st.set_page_config(page_title="Voice Controlled Video Player", layout="wide")
    st.title("🎤 Voice-Activated Video Player")
    
    # Show debug tools if enabled
    if st.secrets.get("DEBUG_MODE", False):
        debug_page()
        return

    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Record Audio")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#ff0000",
            neutral_color="#006400",
            icon_name="microphone",
            sample_rate=16_000
        )

    with col2:
        st.header("Upload Audio")
        audio_file = st.file_uploader("Choose WAV file", type=["wav"])

    # Process inputs
    result_text = None
    if audio_bytes:
        with st.spinner("Processing recording..."):
            result_text = process_audio(audio_bytes)
    
    if audio_file:
        with st.spinner("Processing file..."):
            result_text = process_audio(audio_file.read())

    # Show results
    if result_text:
        show_results(result_text)

def show_results(text):
    st.subheader("Results")
    st.markdown(f"**Recognized command:** `{text}`")
    
    if text in VIDEO_MAPPING:
        video_path = VIDEO_MAPPING[text]
        try:
            st.video(str(video_path))
            st.success(f"Playing: {text}")
        except Exception as e:
            st.error(f"Error playing video: {str(e)}")
    else:
        st.warning(f"No video mapped for: {text}")
        st.info("Available commands: " + ", ".join(VIDEO_MAPPING.keys()))

if __name__ == "__main__":
    main()
