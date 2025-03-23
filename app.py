import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import whisper
import os


@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

def record_audio(duration=5, samplerate=44100):
    st.write("Recording... Speak now!")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    st.write("Recording finished.")
    return audio_data, samplerate

def save_audio(filename, audio_data, samplerate):
    wavio.write(filename, audio_data, samplerate, sampwidth=2)

def transcribe_audio(audio_path):
    st.write("Transcribing audio...")
    result = model.transcribe(audio_path)
    return result["text"]

st.title("🎙️ Real-time Speech-to-Text with Whisper")

st.write("This app allows you to record audio or upload a file for transcription.")


if st.button("🎤 Record Audio"):
    audio, sr = record_audio(duration=5)
    audio_path = "recorded_audio.wav"
    save_audio(audio_path, audio, sr)
    st.audio(audio_path, format="audio/wav")

    transcription = transcribe_audio(audio_path)
    st.subheader("📝 Transcription:")
    st.write(transcription)


uploaded_file = st.file_uploader("📂 Upload an audio file (WAV, MP3, etc.)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(audio_path, format="audio/wav")


    transcription = transcribe_audio(audio_path)
    st.subheader("📝 Transcription:")
    st.write(transcription)


st.write("🚀 Powered by OpenAI Whisper & Streamlit")
