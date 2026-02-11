import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Voice Emotion Analysis", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f4f7fc;
}

/* Title Styling */
h1 {
    text-align: center;
    color: #1f2937;
    font-size: 42px !important;
    font-weight: 700;
}

/* Section Headings */
h2, h3 {
    color: #1e3a8a;
    font-weight: 600;
}

/* File uploader card */
section[data-testid="stFileUploader"] {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

/* Main container spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Success / Alert Styling */
div[data-testid="stAlert"] {
    border-radius: 15px !important;
    font-size: 18px;
    font-weight: 600;
}

/* Feature Box Look */
.stMarkdown {
    font-size: 16px;
}

/* Button Styling */
button[kind="primary"] {
    background-color: #2563eb !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------
st.markdown("<h1>🎤 Voice Emotion Analysis Dashboard</h1>", unsafe_allow_html=True)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.success("Audio uploaded successfully!")

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)

        # ---------------- FEATURE EXTRACTION ----------------
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        st.subheader("🔎 Extracted Features")
        st.write(f"RMS Energy: {rms:.5f}")
        st.write(f"Zero Crossing Rate: {zcr:.5f}")
        st.write(f"Spectral Centroid: {spectral_centroid:.2f}")

        # ---------------- IMPROVED EMOTION LOGIC ----------------
        if rms > 0.04 and spectral_centroid > 1500:
            emotion = "Positive"
        elif rms < 0.02:
            emotion = "Neutral"
        else:
            emotion = "Negative"

        st.subheader("🎯 Detected Emotion")

        if emotion == "Positive":
            st.success("😊 Positive Emotion Detected")
        elif emotion == "Negative":
            st.error("😔 Negative Emotion Detected")
        else:
            st.info("😐 Neutral Emotion Detected")

        # ---------------- WAVEFORM ----------------
        st.subheader("📊 Audio Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Waveform")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    except Exception as e:
        st.error("Error processing audio file.")
        st.write(e)

    finally:
        os.remove(file_path)
