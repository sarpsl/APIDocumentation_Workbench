
import os
import json
import torch
import optuna
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Paths to data
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/train_cleaned_declarative.json'))
val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/validation_cleaned_declarative.json'))
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/test_cleaned_declarative.json'))

# Model name (Hugging Face Transformers compatible)
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Load dataset

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_dataset(path):
    data = load_jsonl(path)
    # Format: code as input, doc as target
    texts = []
    for item in data:
        code = item.get('code', None)
        doc = item.get('doc', None)
        if code is not None and doc is not None:
            code_len = len(code)
            doc_len = len(doc)
            if 15 <= code_len <= 220 and 10 <= doc_len <= 120:
                texts.append(f"<code>\n{code}\n<doc>\n{doc}")
    return Dataset.from_dict({'text': texts})


train_dataset = prepare_dataset(train_path)
val_dataset = prepare_dataset(val_path)
test_dataset = prepare_dataset(test_path)

# Limit to 500 training and 50 validation records
if len(train_dataset) > 500:
    train_dataset = train_dataset.select(range(500))
if len(val_dataset) > 50:
    val_dataset = val_dataset.select(range(50))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

def model_init():
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Batch size: {per_device_train_batch_size}")
    steps_per_epoch = (len(train_dataset) + per_device_train_batch_size - 1) // per_device_train_batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps (all epochs): {steps_per_epoch * num_train_epochs}")

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        fp16=True,
        bf16=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    model = model_init()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return eval_result["eval_loss"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    print(study.best_trial)
    # Save the best model
    best_model = model_init()
    training_args = TrainingArguments(output_dir="./results_final", num_train_epochs=study.best_trial.params["num_train_epochs"], per_device_train_batch_size=study.best_trial.params["per_device_train_batch_size"], learning_rate=study.best_trial.params["learning_rate"], report_to="none", fp16=True, bf16=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=best_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    # Save to models/finetuned directory in project root
    finetuned_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/finetuned'))
    os.makedirs(finetuned_dir, exist_ok=True)
    best_model.save_pretrained(finetuned_dir)
    tokenizer.save_pretrained(finetuned_dir)
