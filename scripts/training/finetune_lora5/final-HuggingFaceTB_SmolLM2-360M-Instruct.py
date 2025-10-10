from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import torch
import os
import torch

# Monkey-patch MatmulLtState to always have memory_efficient_backward attribute
try:
    import bitsandbytes.autograd._functions
    if not hasattr(bitsandbytes.autograd._functions.MatmulLtState, "memory_efficient_backward"):
        setattr(bitsandbytes.autograd._functions.MatmulLtState, "memory_efficient_backward", False)
except Exception as e:
    print(f"Warning: Could not patch MatmulLtState: {e}")

# ==== Step 1: Load and format your dataset ====
with open("data/train_cleaned_declarative-10000.json") as f:
    train_data = json.load(f)
with open("data/validation_cleaned_declarative.json") as f:
    val_data = json.load(f)

# Assume each entry has "code" and "doc"
def format_data_item(item):
    return {
        "prompt": f"{item['code']}",
        "output": item['doc']
    }

def prepare_dataset(data, n=100):
    filtered = []
    for item in data:
        code = item.get('code', None)
        doc = item.get('doc', None)
        if code is not None and doc is not None:            
            filtered.append(item)
            if len(filtered) >= n:
                break
            # code_len = len(code)
            # doc_len = len(doc)
            # if 15 <= code_len <= 220 and 10 <= doc_len <= 120:
            #     filtered.append(item)
            #     if len(filtered) >= n:
            #         break
    return filtered

train_dataset = prepare_dataset(train_data, 10000)
train_dataset = Dataset.from_list(train_dataset)
train_dataset = train_dataset.map(format_data_item)

val_dataset = prepare_dataset(val_data, 1000)
val_dataset = Dataset.from_list(val_dataset)
val_dataset = val_dataset.map(format_data_item)

# ==== Step 2: Load tokenizer and model ====
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    load_in_8bit_fp32_cpu_offload=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# ==== Step 4: Tokenize ====
def tokenize_and_prepare(example):
    prompt_text = f"Generate documentation for the following code:\n{example['prompt']}\n\nDocumentation:\n"
    target_text = example['output']
    full_text = prompt_text + target_text

    # Tokenize
    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Convert input_ids to tensor
    input_ids = torch.tensor(tokenized["input_ids"])
    
    # Mask prompt tokens
    prompt_len = len(tokenizer(prompt_text)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    tokenized["labels"] = labels.tolist()  # convert back to list for HF Dataset
    return tokenized

tokenized_train_dataset = train_dataset.map(
    tokenize_and_prepare, batched=False, remove_columns=train_dataset.column_names
)
tokenized_val_dataset = val_dataset.map(
    tokenize_and_prepare, batched=False, remove_columns=val_dataset.column_names
)

# ==== Print Dataset Information ====
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Training dataset size: {len(tokenized_train_dataset)} records")
print(f"Validation dataset size: {len(tokenized_val_dataset)} records")
print()

# Print sample training records
print("SAMPLE TRAINING RECORDS:")
print("-" * 40)
for i, example in enumerate(train_dataset.select(range(min(3, len(train_dataset))))):
    print(f"Training Record {i+1}:")
    print(f"  Code: {example['prompt'][:100]}{'...' if len(example['prompt']) > 100 else ''}")
    print(f"  Documentation: {example['output'][:100]}{'...' if len(example['output']) > 100 else ''}")
    print()

# Print sample validation records
print("SAMPLE VALIDATION RECORDS:")
print("-" * 40)
for i, example in enumerate(val_dataset.select(range(min(3, len(val_dataset))))):
    print(f"Validation Record {i+1}:")
    print(f"  Code: {example['prompt'][:100]}{'...' if len(example['prompt']) > 100 else ''}")
    print(f"  Documentation: {example['output'][:100]}{'...' if len(example['output']) > 100 else ''}")
    print()

# Print tokenization statistics
train_lengths = [len(tokenizer.decode(example['input_ids'], skip_special_tokens=True)) for example in tokenized_train_dataset.select(range(min(100, len(tokenized_train_dataset))))]
val_lengths = [len(tokenizer.decode(example['input_ids'], skip_special_tokens=True)) for example in tokenized_val_dataset.select(range(min(100, len(tokenized_val_dataset))))]

print("TOKENIZATION STATISTICS:")
print("-" * 40)
print(f"Training - Average text length: {sum(train_lengths)/len(train_lengths):.1f} characters")
print(f"Training - Min/Max text length: {min(train_lengths)}/{max(train_lengths)} characters")
print(f"Validation - Average text length: {sum(val_lengths)/len(val_lengths):.1f} characters")
print(f"Validation - Min/Max text length: {min(val_lengths)}/{max(val_lengths)} characters")
print()

print("Starting training in 3 seconds...")

# Workaround for missing memory_efficient_backward attribute in MatmulLtState
for module in model.modules():
    state = getattr(module, "state", None)
    if state is not None and not hasattr(state, "memory_efficient_backward"):
        state.memory_efficient_backward = False
import time
time.sleep(3)
print("=" * 60)

# ==== Step 5: Training Arguments ====
output_dir = os.path.join("models", model_name.replace("/", "_"))
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    num_train_epochs=2,
    learning_rate=1e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    remove_unused_columns=False
)

# ==== Step 6: Trainer Setup ====
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator
)

# ==== Step 7: Train ====
# Resume from checkpoint if it exists
checkpoint_path = os.path.join(output_dir, "checkpoint-125")
if os.path.exists(checkpoint_path):
    print(f"Resuming training from {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("No checkpoint found, starting fresh training")
    trainer.train()

# ==== Step 8: Save final model ====
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
