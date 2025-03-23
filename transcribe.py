import whisper
from transformers import pipeline


model = whisper.load_model("base") 


audio_path = "C:/Users/soham kumar/Desktop/new ai/sample/Riya.mp3"


result = model.transcribe(audio_path)
transcription = result["text"]

print("\n🎤 Audio File:", audio_path)
print("📝 Original Transcription:\n", transcription)


corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")


corrected_transcription = corrector(transcription, max_length=512)[0]['generated_text']

print("\n✅ Corrected Transcription:\n", corrected_transcription)
