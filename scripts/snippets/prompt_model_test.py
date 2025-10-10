from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "BEE-spoke-data/smol_llama-81M-tied"
# model_name = "HuggingFaceTB/SmolLM-360M-Instruct-WebGPU"
# model_name = "HuggingFaceTB/SmolLM2-360M"
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(model_name)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def get_response(prompt: str) -> str:
    # Actual user prompt
    messages = [
        {"role": "user", "content": "Transform imperative to declarative: Write a function to split a string into parts."},
        {"role": "assistant", "content": "The function returns an array of strings."},
        {"role": "user", "content": "Transform imperative to declarative: Write a function to multiply two numbers."},
        {"role": "assistant", "content": "The function returns the multiplication of two numbers."},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
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
        temperature=0.5,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    print("Enter your prompt (type 'exit' to quit):")
    while True:
        prompt = input(">>> ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        result = get_response(prompt)
        print("Result:", result)

