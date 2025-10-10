import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model
import sklearn

# url = "https://raw.githubusercontent.com/hetpandya/paraphrase-datasets-pretrained-models/refs/heads/main/datasets/tapaco/tapaco_paraphrases_dataset.csv"

dataset_df = pd.read_csv("datasets/tapaco/tapaco_paraphrases_dataset.csv",sep="\t+")

# Renaming the columns
dataset_df.columns = ["input_text","target_text"]
# Adding a prefix. Here we shall keep "paraphrase" as a prefix.
dataset_df["prefix"] = "paraphrase"

train_data,test_data = train_test_split(dataset_df,test_size=0.1)

args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "num_train_epochs": 4,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": False,
    "adam_epsilon": 1e-08,
    "eval_batch_size": 6,
    "fp_16": False,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "n_gpu": 1,
    "seed": 42,
    "train_batch_size": 6,
    "warmup_steps": 0,
    "weight_decay": 0.0,
}

model = T5Model("t5","t5-small", args=args)
model.train_model(train_data, eval_data=test_data, use_cuda=True,acc=sklearn.metrics.accuracy_score)
