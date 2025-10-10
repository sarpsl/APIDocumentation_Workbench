from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "BEE-spoke-data/smol_llama-81M-tied"
# model_name = "HuggingFaceTB/SmolLM2-360M"
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "HuggingFaceTB/SmolLM2-1.7B"

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

def get_response_string(prompt: str) -> str:
    # Manually construct the one-shot prompt
    example_1 = "Transform imperative to declarative: Write a function to split a string into parts.\nThe function returns an array of strings."
    example_2 = "Transform imperative to declarative: Write a function to multiply two numbers.\nThe function returns the multiplication of two numbers."
    user_prompt = f"Transform imperative to declarative: {prompt}\n"

    # Combine examples and user prompt
    input_text = f"{example_1}\n{example_2}\n{user_prompt}"

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
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the last line as the answer
    last_reply = [line for line in decoded.split("\n") if line.strip()][-1]
    return last_reply

# Load your dataset
import json
from tqdm import tqdm

with open("datasets/final/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Rewrite documentation
for item in tqdm(data, desc="Rewriting documentation"):
    if "documentation" in item:
        item["fixed_documentation"] = get_response_chat_template(item["documentation"])

# Save the updated dataset
with open("datasets/final/final_dataset_rewritten.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    