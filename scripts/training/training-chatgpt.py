#pip install transformers datasets accelerate evaluate peft

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Salesforce/codet5-base-multi-sum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                # rank
    lora_alpha=32,       # scaling factor
    target_modules=["q", "v"],  # apply to attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.to(0)
model.print_trainable_parameters()

from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "data/codexglue/code-text/python/final/jsonl/train/python_train_0.jsonl",
    "validation": "data/codexglue/code-text/python/final/jsonl/valid/python_valid_0.jsonl",
    "test": "data/codexglue/code-text/python/final/jsonl/test/python_test_0.jsonl"
}, split=None)

max_source_length = 128
max_target_length = 64

def preprocess_function(examples):
    # use "code" as input and "docstring" as target
    model_inputs = tokenizer(
        examples["code"],
        max_length=max_source_length,
        truncation=True,
        padding="max_length",         # optional: helps batching
        text_target=examples["docstring"],
        # max_length_target=max_target_length,
        # truncation_target=True
    )
    return model_inputs

# Apply preprocessing & drop unused metadata
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names
)

from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./codet5-base-multi-sum",
    eval_strategy="steps",      # or "epoch"
    eval_steps=2000,                  # evaluate less frequently
    logging_steps=200,
    learning_rate=1e-4,
    per_device_train_batch_size=2,    # small to fit on T1000
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,    # effective batch size = 16
    num_train_epochs=2,               # start small
    fp16=True,                        # MUST for speed
    dataloader_num_workers=4,
    save_total_limit=2,
    predict_with_generate=True,
    remove_unused_columns=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./codet5-base-multi-sum-lora-adapter")
