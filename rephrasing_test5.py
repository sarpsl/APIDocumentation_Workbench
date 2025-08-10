from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load Mistral Instruct model from Hugging Face (can be local or downloaded)
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Or your local path if downloaded

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def imperative_to_declarative_mistral(imperative: str) -> str:
    prompt = f"""Convert the following imperative sentence into a declarative one. 
It must not contain words like 'should', 'need to', or 'must'. Use a factual tone.

Imperative: Write a function to split a string into parts
Declarative: The function returns an array of strings.

Imperative: {imperative}
Declarative:"""

    response = pipe(prompt, max_new_tokens=60, do_sample=False)[0]['generated_text']
    # Extract only the part after "Declarative:"
    result = response.split("Declarative:")[-1].strip().split('\n')[0]
    return result

# Example
imperative = "Create a method to validate user input"
print("Declarative:", imperative_to_declarative_mistral(imperative))
