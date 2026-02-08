import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile

st.set_page_config(page_title="Voice Emotion Analysis", layout="centered")

st.title("🎤 Voice Emotion Analysis Dashboard")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.success("Audio uploaded successfully!")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load audio safely
    y, sr = librosa.load(file_path, sr=None)

    # Feature extraction
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Simple rule-based emotion
    if rms > 0.05 and zcr > 0.1:
        emotion = "Positive 😊"
    elif rms < 0.03:
        emotion = "Neutral 😐"
    else:
        emotion = "Negative 😞"

    st.subheader("🎯 Detected Emotion")
    st.success(emotion)

    # Plot waveform
    st.subheader("📊 Audio Waveform")
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    st.pyplot(fig)

    os.remove(file_path)
