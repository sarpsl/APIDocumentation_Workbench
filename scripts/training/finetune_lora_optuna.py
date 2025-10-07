
import os
import json
import torch
from datasets import Dataset
import numpy as np
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Optuna integration
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import threading

# ==== Set random seed for reproducibility ====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
# ==== Step 1: Load and format your dataset ====
with open("data/train_cleaned_declarative.json") as f:
    train_data = json.load(f)
with open("data/validation_cleaned_declarative.json") as f:
    val_data = json.load(f)

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

training_records = 1000

train_dataset = prepare_dataset(train_data, training_records)
train_dataset = Dataset.from_list(train_dataset)
train_dataset = train_dataset.map(format_data_item)

val_dataset = prepare_dataset(val_data, training_records // 10)
val_dataset = Dataset.from_list(val_dataset)
val_dataset = val_dataset.map(format_data_item)


# ==== Optuna Objective Function ====
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 4, 8])
    per_device_eval_batch_size = trial.suggest_categorical('per_device_eval_batch_size', [2, 4, 8])
    lora_r = trial.suggest_categorical('lora_r', [8, 16, 32])
    lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
    lora_dropout = trial.suggest_float('lora_dropout', 0.01, 0.2)
    num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
    max_length = trial.suggest_categorical('max_length', [256, 384, 512])

    # Model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)

    def tokenize_and_prepare(example):
        tokens = tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_train_dataset = train_dataset.map(
        tokenize_and_prepare, batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_and_prepare, batched=True, remove_columns=val_dataset.column_names
    )

    output_dir = os.path.join("models", f"optuna_trial_{trial.number}")
    os.makedirs(output_dir, exist_ok=True)
    # Use bf16 if available, else fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",  # Save at end of each epoch for safety
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_dir="./logs",
        logging_steps=20,  # Log less frequently for larger datasets
        save_total_limit=2,
        remove_unused_columns=True,
        report_to=["tensorboard"],  # Enable TensorBoard logging
        disable_tqdm=False
    )
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
    trainer.train()
    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics["eval_loss"]
    # Log all params and metrics for dashboard/thesis
    trial.set_user_attr("learning_rate", learning_rate)
    trial.set_user_attr("per_device_train_batch_size", per_device_train_batch_size)
    trial.set_user_attr("per_device_eval_batch_size", per_device_eval_batch_size)
    trial.set_user_attr("lora_r", lora_r)
    trial.set_user_attr("lora_alpha", lora_alpha)
    trial.set_user_attr("lora_dropout", lora_dropout)
    trial.set_user_attr("num_train_epochs", num_train_epochs)
    trial.set_user_attr("max_length", max_length)
    trial.set_user_attr("val_loss", val_loss)
    return val_loss

# ==== Optuna Study Setup ====
def run_optuna():
    storage = "sqlite:///optuna_qlora.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="qlora_finetune",
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    print(study.best_trial)
    # Save best model config for reproducibility
    with open("best_trial_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print("Optuna study complete. To view dashboard, run: optuna dashboard optuna_qlora.db")

# ==== Optuna Dashboard Thread ====
def launch_dashboard():
    import subprocess
    subprocess.run(["optuna-dashboard", "sqlite:///optuna_qlora.db", "--port", "8080"])

if __name__ == "__main__":
    # Start dashboard in background thread
    # dashboard_thread = threading.Thread(target=launch_dashboard, daemon=True)
    # dashboard_thread.start()
    run_optuna()
