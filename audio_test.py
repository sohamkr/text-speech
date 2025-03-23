import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

audio_path = "C:/Users/soham kumar/Desktop/new ai/audio_flac.mp3"
audio, sr = librosa.load(audio_path, sr=16000)


input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)


with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
