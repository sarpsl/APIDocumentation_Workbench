from simpletransformers.t5 import T5Model
import os

root_dir = os.getcwd()

trained_model_path = os.path.join(root_dir,"outputs")

args = {
"overwrite_output_dir": True,
"max_seq_length": 256,
"max_length": 50,
"top_k": 50,
"top_p": 0.95,
"num_return_sequences": 5
}

trained_model = T5Model("t5",trained_model_path,args=args)

prefix = "paraphrase"
pred = trained_model.predict([f"{prefix}: Write a function to add two numbers"])
print("\n")
print(pred)
