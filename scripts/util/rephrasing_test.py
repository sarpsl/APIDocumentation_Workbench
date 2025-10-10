from transformers import pipeline

# Load your dataset
import json

with open("datasets/final/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Load a local language model pipeline (make sure you have the model downloaded)
paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def rewrite_doc(doc):
    prompt = f"Paraphrase: {doc}"
    result = paraphraser(prompt, max_length=64, num_return_sequences=1)
    return result[0]['generated_text']

str = "Write a function that takes a string and returns it in uppercase."
print(rewrite_doc(str))
