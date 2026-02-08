import streamlit as st
import os
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
import matplotlib.pyplot as plt

UPLOAD_FOLDER = "uploads"
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

st.set_page_config(page_title="Voice Emotion Analyzer", layout="centered")
st.title("🎤 Voice Emotion Analysis Dashboard")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Audio uploaded successfully!")

    audio = AudioSegment.from_file(file_path)
    recognizer = sr.Recognizer()

    times = []
    emotions = []

    chunk_length = 5000

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_path = os.path.join(UPLOAD_FOLDER, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                emotion = detect_emotion(text)
            except:
                emotion = "Unrecognized"

        times.append(f"{i//1000}s")
        emotions.append(emotion)
        os.remove(chunk_path)

    st.subheader("📊 Emotion Timeline")

    for t, e in zip(times, emotions):
        st.write(f"**{t} → {e}**")

    emotion_map = {"Negative": 0, "Neutral": 1, "Positive": 2, "Unrecognized": 1}
    numeric = [emotion_map[e] for e in emotions]

    fig, ax = plt.subplots()
    ax.plot(times, numeric, marker="o")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Emotion")
    ax.set_title("Emotion Over Time")

    st.pyplot(fig)
