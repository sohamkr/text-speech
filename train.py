import torch
from datasets import load_dataset, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer

dataset = load_dataset("csv", data_files={"train": "British English Speech Recognition.csv"})


if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2)
    print("No test split found. Splitting train dataset into train & test...")


dataset = dataset.rename_columns({
    "Audio": "input_features",  
    "Text": "labels"           
})


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def prepare_dataset(batch):
    audio = batch["input_features"] 
    text = batch["labels"]  

    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    batch["input_features"] = inputs.input_features[0] 
    batch["labels"] = processor.tokenizer(text).input_ids  

    return batch


dataset = dataset.map(prepare_dataset, remove_columns=["ID"])


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
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
