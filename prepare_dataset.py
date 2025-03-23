import torch
import librosa  
from datasets import load_dataset, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer

dataset = load_dataset("csv", data_files={"train": "dataset.csv"})

if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2)
    print("No test split found. Splitting train dataset into train & test...")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def prepare_dataset(batch):
    audio_path = batch["Audio"]  

    try:
        audio, sr = librosa.load(audio_path, sr=16000)  

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        batch["input_features"] = inputs.input_features[0]  
        batch["labels"] = processor.tokenizer(batch["Text"]).input_ids  

        return batch

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None 


processed_dataset = dataset.map(prepare_dataset, remove_columns=["ID"])
filtered_dataset = {split: data.filter(lambda x: x is not None) for split, data in processed_dataset.items()}


training_args = TrainingArguments(
    output_dir="./whisper_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none", 
    remove_unused_columns=False  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=filtered_dataset["train"],
    eval_dataset=filtered_dataset["test"]
)


trainer.train()
