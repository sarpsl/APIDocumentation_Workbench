import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================
# Sample Data from Your Log
# =============================
# Training log per step
train_steps = list(range(1, 751))  # 750 training steps
train_loss = [2.0531, 1.9042, 1.8983, 1.7477, 1.7482, 1.7639, 1.637, 1.7183, 1.6125, 1.6593] * 75  # extend to 750
grad_norm = [0.3112, 0.2012, 0.2947, 0.2889, 0.3215, 0.5846, 0.3346, 0.2785, 0.3689, 0.2811] * 75
learning_rate = np.linspace(0.0001976, 2.6667e-07, 750)

# Eval checkpoints (at end of each epoch)
eval_epochs = [1, 2, 3]
eval_loss = [1.4357, 1.4486, 1.4577]

# =============================
# Plot 1: Training vs Eval Loss
# =============================
plt.figure(figsize=(8,5))
plt.plot(train_steps, train_loss, label="Training Loss", color="blue")
for e, l in zip(eval_epochs, eval_loss):
    plt.hlines(l, (e-1)*250, e*250, colors='red', linestyles='dashed', label=f"Eval Loss (Epoch {e})" if e==1 else "")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================
# Plot 2: Gradient Norms over Training
# ====================================
plt.figure(figsize=(8,5))
plt.plot(train_steps, grad_norm, color="green")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms over Training Steps")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# Plot 3: Learning Rate Decay
# ============================
plt.figure(figsize=(8,5))
plt.plot(train_steps, learning_rate, color="purple")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Decay over Training Steps")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# Plot 4: Dataset Text Length Distribution
# ==========================================
# Sample lengths
train_lengths = np.random.normal(783.5, 200, 1000).clip(min=100)  # simulated based on log
val_lengths = np.random.normal(971.4, 300, 100).clip(min=100)

plt.figure(figsize=(8,5))
sns.histplot(train_lengths, color="blue", label="Train", kde=True, bins=30, alpha=0.6)
sns.histplot(val_lengths, color="red", label="Validation", kde=True, bins=30, alpha=0.6)
plt.xlabel("Text Length (chars)")
plt.ylabel("Frequency")
plt.title("Text Length Distribution of Dataset")
plt.legend()
plt.tight_layout()
plt.show()

# ===================================================
# Plot 5: Training + Eval Loss + Final Validation
# ===================================================
plt.figure(figsize=(8,5))
plt.plot(train_steps, train_loss, label="Training Loss", color="blue")
for e, l in zip(eval_epochs, eval_loss):
    plt.scatter([e*250], [l], color='red', s=50, zorder=5, label=f"Eval Loss (Epoch {e})" if e==1 else "")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training + Evaluation Loss with Final Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
