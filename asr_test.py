import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded successfully on:", device)
