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

# Rewrite documentation
for item in data:
    if "documentation" in item:
        item["documentation"] = rewrite_doc(item["documentation"])

# Save the updated dataset
with open("datasets/final/final_dataset_rewritten.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    