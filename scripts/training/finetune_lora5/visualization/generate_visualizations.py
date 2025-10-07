import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
})
# ======== Load JSONL Training Log ========

# log_file = "scripts/training/finetune_lora5/visualization/smollm2_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/qwen-2.5_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/codet5_run_log.jsonl"
# log_file = "scripts/training/finetune_lora5/visualization/flant5_run_log.jsonl"
log_file = "scripts/training/finetune_lora5/visualization/qwen-2.5_jarvislabs_run_log.jsonl"

import os
output_dir = log_file.replace('.jsonl', '')
os.makedirs(output_dir, exist_ok=True)

train_steps, train_loss, grad_norm, learning_rate = [], [], [], []
train_epochs, train_loss_by_epoch = [], []
eval_epochs, eval_loss = [], []
eval_runtime, eval_samples_per_sec, eval_steps_per_sec = [], [], []
train_lengths, val_lengths = [], []
final_train_metrics = {}

with open(log_file, "r") as f:
    for i, line in enumerate(f):
        # Convert single quotes to double quotes and strip whitespace
        line_json = line.strip().replace("'", '"')
        line_json = re.sub(r':\s*nan', ': null', line_json)
        if not line_json:
            continue
        try:
            record = json.loads(line_json)
        except json.JSONDecodeError:
            continue
        if "loss" in record:
            train_steps.append(i+1)
            train_loss.append(record["loss"])
            grad = record.get("grad_norm", 0)
            if grad is None:
                grad = 0
            grad_norm.append(grad)
            learning_rate.append(record.get("learning_rate", np.nan))
            # For epoch-wise aggregation
            train_epochs.append(record["epoch"])
            train_loss_by_epoch.append(record["loss"])
        elif "eval_loss" in record:
            eval_epochs.append(int(record["epoch"]))
            eval_loss.append(record["eval_loss"])
            eval_runtime.append(record.get("eval_runtime", np.nan))
            eval_samples_per_sec.append(record.get("eval_samples_per_second", np.nan))
            eval_steps_per_sec.append(record.get("eval_steps_per_second", np.nan))
        if "train_text_length" in record:
            train_lengths.append(record["train_text_length"])
        if "val_text_length" in record:
            val_lengths.append(record["val_text_length"])
        if "train_runtime" in record:
            final_train_metrics = record



# ======== Plot 1: Training vs Eval Loss ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=train_loss, label="Training Loss", color="#006400")  # dark green
for e, l in zip(eval_epochs, eval_loss):
    plt.hlines(l, (e-1)*len(train_steps)//max(eval_epochs), e*len(train_steps)//max(eval_epochs), colors='#8B0000', linestyles='dashed', label=f"Eval Loss (Epoch {e})" if e==eval_epochs[0] else "")  # dark red
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_vs_eval_loss.png"))
plt.close()

# ======== Plot 2: Training Loss vs. Epoch (Averaged) ========
import pandas as pd
df_train = pd.DataFrame({'epoch': train_epochs, 'loss': train_loss_by_epoch})
avg_train_loss = df_train.groupby('epoch').mean().reset_index()
plt.figure(figsize=(8,5))
sns.lineplot(x=avg_train_loss['epoch'], y=avg_train_loss['loss'], marker='o', label='Avg Training Loss per Epoch', color='#006400')
plt.xlabel('Epoch')
plt.ylabel('Avg Training Loss')
plt.title('Average Training Loss per Epoch')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'avg_training_loss_per_epoch.png'))
plt.close()

# ======== Plot 3: Eval Loss vs. Epoch ========
plt.figure(figsize=(8,5))
sns.lineplot(x=eval_epochs, y=eval_loss, marker='o', label='Eval Loss', color='#8B0000')
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss per Epoch')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eval_loss_per_epoch.png'))
plt.close()

# ======== Plot 4: Eval Runtime vs. Epoch ========
if eval_runtime:
    plt.figure(figsize=(8,5))
    sns.lineplot(x=eval_epochs, y=eval_runtime, marker='o', color='#191970')  # midnight blue
    plt.xlabel('Epoch')
    plt.ylabel('Eval Runtime (s)')
    plt.title('Evaluation Runtime per Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_runtime_per_epoch.png'))
    plt.close()

# ======== Plot 5: Eval Samples/sec and Steps/sec vs. Epoch ========
if eval_samples_per_sec and eval_steps_per_sec:
    plt.figure(figsize=(8,5))
    sns.lineplot(x=eval_epochs, y=eval_samples_per_sec, marker='o', label='Eval Samples/sec', color='#228B22')  # forest green
    sns.lineplot(x=eval_epochs, y=eval_steps_per_sec, marker='o', label='Eval Steps/sec', color='#8B4513')  # saddle brown
    plt.xlabel('Epoch')
    plt.ylabel('Throughput')
    plt.title('Evaluation Throughput per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_throughput_per_epoch.png'))
    plt.close()

# ======== Plot 6: Final Training Summary Bar Plot ========
if final_train_metrics:
    plt.figure(figsize=(8,5))
    metrics = ['train_runtime', 'train_samples_per_second', 'train_steps_per_second', 'train_loss']
    values = [final_train_metrics.get(m, 0) for m in metrics]
    sns.barplot(x=metrics, y=values, palette='dark')
    plt.title('Final Training Summary Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_training_summary.png'))
    plt.close()

# ======== Plot 7: Learning Rate vs. Epoch ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_epochs, y=learning_rate, color='#191970')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_rate_vs_epoch.png'))
plt.close()

# ======== Plot 8: Gradient Norm vs. Epoch ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_epochs, y=grad_norm, color='#228B22')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm over Epochs')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'grad_norm_vs_epoch.png'))
plt.close()

# ======== Plot 9: Train vs. Eval Loss Overlay ========
plt.figure(figsize=(8,5))
sns.lineplot(x=avg_train_loss['epoch'], y=avg_train_loss['loss'], marker='o', label='Avg Training Loss', color='#006400')
sns.lineplot(x=eval_epochs, y=eval_loss, marker='o', label='Eval Loss', color='#8B0000')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs. Eval Loss per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_vs_eval_loss_overlay.png'))
plt.close()

# ======== Plot 10: Best Epoch Marker on Eval Loss Plot ========
if eval_epochs and eval_loss:
    best_idx = int(np.argmin(eval_loss))
    best_epoch = eval_epochs[best_idx]
    best_loss = eval_loss[best_idx]
    plt.figure(figsize=(8,5))
    sns.lineplot(x=eval_epochs, y=eval_loss, marker='o', color='#8B0000')
    plt.scatter([best_epoch], [best_loss], color='#FFD700', s=100, label=f'Best Epoch {best_epoch} (Loss={best_loss:.4f})')  # gold
    plt.xlabel('Epoch')
    plt.ylabel('Eval Loss')
    plt.title('Best Epoch on Eval Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_epoch_eval_loss.png'))
    plt.close()


# ======== Plot 11: Gradient Norms ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=grad_norm, color="green")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms over Training Steps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gradient_norms.png"))
plt.close()



# ======== Plot 12: Learning Rate Decay ========
plt.figure(figsize=(8,5))
sns.lineplot(x=train_steps, y=learning_rate, color="purple")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Decay over Training Steps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_rate_decay.png"))
plt.close()



# ======== Plot 13: Text Length Distribution ========
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




# ======== Plot 14: Training + Eval Loss ========
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