from transformers import 
from transformers import (
    RobertaTokenizer, 
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import torch

# ==== Step 1: Load and format your dataset ====
raw_data = []
with open("data/codexglue/code-text/python/final/jsonl/train/python_train_0.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        raw_data.append(json.loads(line))

# Assume each entry has "code" and "doc"
def format_example(example):
    return {
        "prompt": f"### Instruction:\nGenerate documentation for the following code:\n{example['code']}\n\n### Response:\n{example['doc']}"
    }

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_example)
dataset = dataset.train_test_split(test_size=0.1)

# ==== Step 2: Load tokenizer and model ====
# Use the original Qwen2.5-Coder model (not GGUF version)
# model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model_name = "Salesforce/codet5-base-multi-sum"
# model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "microsoft/DialoGPT-small"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    load_in_8bit_fp32_cpu_offload=True
)

tokenizer = RobertaTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = T5ForConditionalGeneration.from_pretrained(
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

tokenized_dataset = dataset.map(tokenize_and_prepare, batched=True, remove_columns=dataset["train"].column_names)

# ==== Step 5: Training Arguments ====
# output_dir = "./smollm2-docgen"
# output_dir = "./qwen-coder-docgen"
# output_dir = "./smollm2-1.7b"
output_dir = "./codet5-base-multi-sum"
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
    save_total_limit=2
)

# ==== Step 6: Trainer Setup ====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# ==== Step 7: Train ====
trainer.train()

# ==== Step 8: Save final model ====
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
