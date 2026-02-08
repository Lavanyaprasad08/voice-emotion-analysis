from flask import Flask, render_template, request
import os
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Windows-safe folder creation
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def process_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    recognizer = sr.Recognizer()

    chunk_length = 5000  # 5 seconds
    times = []
    emotions = []

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

        times.append(f"{i // 1000}s")
        emotions.append(emotion)

        os.remove(chunk_path)

    return times, emotions


def generate_chart(times, emotions):
    emotion_map = {"Negative": 0, "Neutral": 1, "Positive": 2, "Unrecognized": 1}
    numeric = [emotion_map[e] for e in emotions]

    plt.figure(figsize=(8, 4))
    plt.plot(times, numeric, marker="o")
    plt.yticks([0, 1, 2], ["Negative", "Neutral", "Positive"])
    plt.xlabel("Time")
    plt.ylabel("Emotion")
    plt.title("Emotion Over Time")
    plt.tight_layout()

    chart_path = "static/emotion_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return chart_path


@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    chart = None

    if request.method == "POST":
        audio_file = request.files["audio"]
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        times, emotions = process_audio(file_path)
        chart = generate_chart(times, emotions)

        # FIX: prepare zip in Python, not Jinja
        data = list(zip(times, emotions))

    return render_template(
        "index.html",
        data=data,
        chart=chart
    )


if __name__ == "__main__":
    app.run(debug=True)
