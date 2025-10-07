import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# =========================
# Simulated Data from Log
# =========================
epochs = np.linspace(0, 3, 31)  # 0 to 3 epochs, ~0.1 step increments
train_loss = [6.22, 4.71, 3.38, 2.41, 1.90, 1.74, 1.51, 1.73, 1.27, 1.57,
              1.42, 1.53, 1.17, 1.56, 1.43, 1.56, 1.46, 1.61, 1.56, 1.19,
              1.20, 1.18, 1.57, 1.02, 1.37, 1.12, 1.35, 0.90, 1.14, 1.41, 1.23]
eval_loss = [0.535, 0.509, 0.493]  # epoch 1, 2, 3
eval_epochs = [1, 2, 3]

grad_norm = [2.73, np.nan, 1.53, 1.44, 1.15, 1.19, 0.53, 0.68, 0.34, 0.46,
             0.59, 0.39, 0.49, 0.49, 0.73, 0.71, 0.76, 0.43, 0.77, 0.59,
             0.55, 0.46, 0.85, 0.69, 0.52, 0.80, 0.70, 1.30, 0.77, 0.66, 0.68]

learning_rate = np.linspace(0.0001976, 2.667e-7, len(epochs))

# =========================
# Plot 1: Training vs Eval Loss
# =========================
plt.figure(figsize=(7,5))
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.scatter(eval_epochs, eval_loss, color='red', label="Validation Loss", s=70)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Plot 2: Gradient Norms over Training Steps
# =========================
plt.figure(figsize=(7,5))
plt.plot(epochs, grad_norm, marker='s', color='purple')
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm over Training Steps")
plt.tight_layout()
plt.show()

# =========================
# Plot 3: Learning Rate Decay
# =========================
plt.figure(figsize=(7,5))
plt.plot(epochs, learning_rate, marker='o', color='green')
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Learning Rate (log scale)")
plt.title("Linear Learning Rate Decay")
plt.tight_layout()
plt.show()

# =========================
# Plot 4: Dataset Text Length Distribution
# =========================
train_lengths = np.random.randint(180, 1636, size=1000)
val_lengths = np.random.randint(172, 1917, size=200)

plt.figure(figsize=(7,5))
sns.histplot(train_lengths, color='blue', label='Training', kde=True, stat='density', bins=30)
sns.histplot(val_lengths, color='orange', label='Validation', kde=True, stat='density', bins=30)
plt.xlabel("Text Length (characters)")
plt.ylabel("Density")
plt.title("Text Length Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Plot 5: Training + Evaluation Loss + Final Validation Loss
# =========================

# Final validation loss
final_val_loss = eval_loss[-1]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='blue', linewidth=2)

# Scatter eval loss at checkpoints
plt.scatter(eval_epochs, eval_loss, color='red', s=80, zorder=5, label='Evaluation Loss')

# Annotate eval points
for x, y in zip(eval_epochs, eval_loss):
    plt.text(x + 0.05, y + 0.02, f'{y:.3f}', color='red', fontsize=10)

# Add final validation loss as dashed horizontal line
plt.axhline(y=final_val_loss, color='green', linestyle='--', linewidth=2, label=f'Final Validation Loss ({final_val_loss:.3f})')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss with Evaluation Checkpoints and Final Validation Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
