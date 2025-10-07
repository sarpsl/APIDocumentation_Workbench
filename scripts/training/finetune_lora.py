from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import torch
import os

# ==== Step 1: Load and format your dataset ====
with open("data/train_cleaned_declarative.json") as f:
    train_data = json.load(f)
with open("data/validation_cleaned_declarative.json") as f:
    val_data = json.load(f)

# Assume each entry has "code" and "doc"
def format_data_item(item):
    return {
        "prompt": f"### Instruction:\nGenerate documentation for the following code:\n{item['code']}\n\n### Response:\n{item['doc']}"
    }

def prepare_dataset(data, n=100):
    filtered = []
    for item in data:
        code = item.get('code', None)
        doc = item.get('doc', None)
        if code is not None and doc is not None:
            code_len = len(code)
            doc_len = len(doc)
            if 15 <= code_len <= 220 and 10 <= doc_len <= 120:
                filtered.append(item)
                if len(filtered) >= n:
                    break
    return filtered

train_dataset = prepare_dataset(train_data, 500)
train_dataset = Dataset.from_list(train_dataset)
train_dataset = train_dataset.map(format_data_item)

val_dataset = prepare_dataset(val_data, 50)
val_dataset = Dataset.from_list(val_dataset)
val_dataset = val_dataset.map(format_data_item)

# ==== Step 2: Load tokenizer and model ====
# Use the original Qwen2.5-Coder model (not GGUF version)
# model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# Alternative smaller models if needed:
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "microsoft/DialoGPT-small"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    load_in_8bit_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# ==== Step 3: Apply PEFT/LoRA ====
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# ==== Step 4: Tokenize ====
def tokenize_and_prepare(example):
    tokens = tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train_dataset = train_dataset.map(
    tokenize_and_prepare, batched=True, remove_columns=train_dataset.column_names
)
tokenized_val_dataset = val_dataset.map(
    tokenize_and_prepare, batched=True, remove_columns=val_dataset.column_names
)

# ==== Step 5: Training Arguments ====
output_dir = os.path.join("models", model_name.replace("/", "_"))
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False
)

# ==== Step 6: Trainer Setup ====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator
)

# ==== Step 7: Train ====
trainer.train()

# ==== Step 8: Save final model ====
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
