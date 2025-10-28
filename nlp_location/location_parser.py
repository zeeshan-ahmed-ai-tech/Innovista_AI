from datasets import Dataset, DatasetDict
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import pandas as pd
import torch

# Step 1: Read your labeled data
def load_data(file_path):
    sentences = []
    tags = []
    tokens = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    tags.append(labels)
                    tokens = []
                    labels = []
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    labels.append(tag)
    return sentences, tags

# Load your data
train_file = "train_sample_100.txt"  # Must be in same directory
sentences, tags = load_data(train_file)

# Label mapping
unique_labels = sorted(set(label for label_list in tags for label in label_list))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Convert tags to IDs
encoded_tags = [[label2id[label] for label in tag_seq] for tag_seq in tags]

# Hugging Face Dataset
dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": encoded_tags})
dataset = DatasetDict({
    "train": dataset,
    "validation": dataset.select(range(10))  # small validation set
})

# Load tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

# Align tokens and labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # subword token
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load model
model = XLMRobertaForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# Data collator (fixes padding issue)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
model.save_pretrained("ner-xlmr-location")
tokenizer.save_pretrained("ner-xlmr-location")

# 🔽 Export sample predictions to Excel (optional)
def predict_sample():
    inputs = tokenizer(sentences[:10], padding=True, truncation=True,
                       is_split_into_words=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    rows = []
    for i, sentence in enumerate(sentences[:10]):
        row = []
        for j, word in enumerate(sentence):
            label_idx = predictions[i][j].item()
            if label_idx != -100:
                row.append((word, id2label.get(label_idx, "O")))
        rows.extend(row)

    df = pd.DataFrame(rows, columns=["Word", "Predicted Label"])
    df.to_excel("predictions_sample.xlsx", index=False)

predict_sample()
