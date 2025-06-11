from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

# Load the tokenizer and model
# model_name = 't5-small'
# few more models to try: 'Vamsi/T5_Paraphrase_Paws', 't5-base', 't5-large'
# model_name = 'Vamsi/T5_Paraphrase_Paws'
# model_name = 't5-base'
model_name = 't5-large'

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# load dataset
with open("datasets/final/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# collect documentation from top 10 items from the dataset
documentation = [item['documentation'] for item in data[:10]]
print (documentation)

def rewrite_doc(doc):
    input_text = f"paraphrase from imperative to declarative for: {doc}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    return paraphrase

# run rewrite_doc on collected items
for doc in documentation:
    paraphrased_doc = rewrite_doc(doc)
    print(f"Original: {doc}")
    print(f"Paraphrase: {paraphrased_doc}\n")

