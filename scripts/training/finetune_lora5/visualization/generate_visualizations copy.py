import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
# ======== Load JSONL Training Log ========

log_file = "scripts/training/finetune_lora5/visualization/smollm2_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/qwen-2.5_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/codet5_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/flant5_run_log.jsonl"

import os
output_dir = log_file.replace('.jsonl', '')
os.makedirs(output_dir, exist_ok=True)

train_steps, train_loss, grad_norm, learning_rate = [], [], [], []
eval_epochs, eval_loss = [], []
train_lengths, val_lengths = [], []

with open(log_file, "r") as f:
    for i, line in enumerate(f):
        # Convert single quotes to double quotes and strip whitespace
        line_json = line.strip().replace("'", '"')
        record = json.loads(line_json)
        if "loss" in record:
            train_steps.append(i+1)
            train_loss.append(record["loss"])
            grad_norm.append(record.get("grad_norm", np.nan))
            learning_rate.append(record.get("learning_rate", np.nan))
        elif "eval_loss" in record:
            eval_epochs.append(int(record["epoch"]))
            eval_loss.append(record["eval_loss"])
        # Optionally, parse text lengths if present
        if "train_text_length" in record:
            train_lengths.append(record["train_text_length"])
        if "val_text_length" in record:
            val_lengths.append(record["val_text_length"])



# ======== Plot 1: Training vs Eval Loss ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=train_loss, label="Training Loss", color="blue")
for e, l in zip(eval_epochs, eval_loss):
    plt.hlines(l, (e-1)*len(train_steps)//max(eval_epochs), e*len(train_steps)//max(eval_epochs), colors='red', linestyles='dashed', label=f"Eval Loss (Epoch {e})" if e==eval_epochs[0] else "")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_vs_eval_loss.png"))
plt.close()



# ======== Plot 2: Gradient Norms ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=grad_norm, color="green")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms over Training Steps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gradient_norms.png"))
plt.close()



# ======== Plot 3: Learning Rate Decay ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=learning_rate, color="purple")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Decay over Training Steps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_rate_decay.png"))
plt.close()



# ======== Plot 4: Text Length Distribution ========
if train_lengths and val_lengths:
    plt.figure(figsize=(8,5))
    sns.histplot(train_lengths, color="blue", label="Train", kde=True, bins=30, alpha=0.6)
    sns.histplot(val_lengths, color="red", label="Validation", kde=True, bins=30, alpha=0.6)
    plt.xlabel("Text Length (chars)")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution of Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
    plt.close()




# ======== Plot 5: Training + Eval Loss ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=train_loss, label="Training Loss", color="blue")
for e, l in zip(eval_epochs, eval_loss):
    sns.scatterplot(x=[e*len(train_steps)//max(eval_epochs)], y=[l], color='red', s=50, zorder=5, label=f"Eval Loss (Epoch {e})" if e==eval_epochs[0] else "")

# Detect and plot final validation loss if present
final_val_loss = None
final_val_step = None
with open(log_file, "r") as f:
    for i, line in enumerate(f):
        line_json = line.strip().replace("'", '"')
        record = json.loads(line_json)
        # Look for a final eval_loss (highest epoch)
        if "eval_loss" in record:
            if final_val_loss is None or record["epoch"] > final_val_step:
                final_val_loss = record["eval_loss"]
                final_val_step = record["epoch"]

if final_val_loss is not None:
    plt.axhline(final_val_loss, color='orange', linestyle='--', label=f'Final Validation Loss ({final_val_loss:.4f})')

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training + Evaluation Loss with Final Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_eval_final_val_loss.png"))
plt.close()