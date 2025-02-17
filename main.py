import streamlit as st
from transformers import pipeline
import numpy as np
import wave
import io
import logging
import pathlib
from audio_recorder_streamlit import audio_recorder

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_MAPPING = {
    "hello": VIDEO_DIR / "hello.mp4",
    "yes": VIDEO_DIR / "yes.mp4",
    "no": VIDEO_DIR / "no.mp4"
}

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    try:
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device_map="auto"
        )
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        st.stop()

def process_wav_bytes(audio_bytes):
    """Convert WAV bytes to numpy array with proper handling"""
    try:
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                samp_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())

        # Convert to numpy array
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        audio_array = np.frombuffer(frames, dtype=dtype_map[samp_width])

        # Convert stereo to mono
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

        return audio_array, framerate
    except Exception as e:
        logger.error(f"WAV processing error: {str(e)}")
        return None, None

def process_audio(audio_bytes):
    try:
        audio_array, sr = process_wav_bytes(audio_bytes)
        if audio_array is None:
            return None
            
        pipe = load_whisper_model()
        result = pipe({
            "raw": audio_array,
            "sampling_rate": sr
        })
        return result["text"].lower()
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        return None

def main():
    st.title("🎤 Voice-Activated Video Player")
    
    # Audio recording
    audio_bytes = audio_recorder()
    if audio_bytes:
        with st.spinner("Processing..."):
            text = process_audio(audio_bytes)
            if text:
                show_results(text)

    # File upload
    audio_file = st.file_uploader("Upload WAV", type=["wav"])
    if audio_file:
        with st.spinner("Processing file..."):
            text = process_audio(audio_file.read())
            if text:
                show_results(text)

def show_results(text):
    st.subheader("Results")
    st.write(f"Recognized: {text}")
    
    if text in VIDEO_MAPPING:
        st.video(str(VIDEO_MAPPING[text]))
    else:
        st.warning("No matching video found")

if __name__ == "__main__":
    main()
