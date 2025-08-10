from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def get_response_chat_template(prompt: str) -> str:
    messages = [
        {"role": "user", "content": "Transform imperative to declarative: Write a function to split a string into parts."},
        {"role": "assistant", "content": "The function returns an array of strings."},
        {"role": "user", "content": "Transform imperative to declarative: Write a function to multiply two numbers."},
        {"role": "assistant", "content": "The function returns the multiplication of two numbers."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the last assistant response
    # Try splitting by "Assistant:" or "\n" if your template doesn't use "Assistant:"
    if "Assistant:" in decoded:
        last_reply = decoded.split("Assistant:")[-1].strip()
    elif "assistant:" in decoded:
        last_reply = decoded.split("assistant:")[-1].strip()
    else:
        # Fallback: get the last non-empty line
        last_reply = [line for line in decoded.split("\n") if line.strip()][-1]
    return last_reply


# Load your dataset
import json

with open("datasets/final/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Rewrite documentation
for item in data[:10]:
    if "documentation" in item:
        doc = item["documentation"]
        result = get_response_chat_template(item["documentation"])
        print(f"\nDocumentation: {doc}, \nResult: {result}")

