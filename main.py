import streamlit as st
from transformers import pipeline
import tempfile
import os
import re
from audio_recorder_streamlit import audio_recorder

# Configuration
VIDEO_MAPPING = {
    "hello": "videos/hello.mp4",
    "yes": "videos/yes.mp4",
    "no": "videos/no.mp4"  # Verify this file exists
}

@st.cache_resource
def load_whisper_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base"
    )

def process_audio(audio_bytes):
    """Handle audio processing with better text normalization"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        pipe = load_whisper_model()
        result = pipe(tmp_path)
        cleaned_text = clean_text(result["text"])
        return cleaned_text
    
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass

def clean_text(text):
    """Normalize text for better matching"""
    # Remove punctuation and make lowercase
    text = re.sub(r'[^\w\s]', '', text).strip().lower()
    # Keep only single words
    return text.split()[0] if text else ""

def main():
    st.title("🎤 Voice-Controlled Video Player")
    
    # Audio recording
    audio_bytes = audio_recorder()
    if audio_bytes:
        with st.spinner("Processing..."):
            text = process_audio(audio_bytes)
            if text:
                show_results(text)

    # File upload
    audio_file = st.file_uploader("Upload audio", type=["wav"])
    if audio_file:
        with st.spinner("Processing..."):
            text = process_audio(audio_file.read())
            if text:
                show_results(text)

def show_results(text):
    st.subheader("Results")
    st.write(f"Recognized: **{text}**")
    
    # Debugging information
    st.write("Available videos:", list(VIDEO_MAPPING.keys()))
    
    # Check file existence
    if text in VIDEO_MAPPING:
        if os.path.exists(VIDEO_MAPPING[text]):
            st.video(VIDEO_MAPPING[text])
            st.success(f"Playing video for: {text}")
        else:
            st.error(f"Video file not found: {VIDEO_MAPPING[text]}")
    else:
        st.warning("No matching video found")

if __name__ == "__main__":
    main()