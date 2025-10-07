from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# model_name = "models/HuggingFaceTB_SmolLM2-1.7B-Instruct"
model_name = "models/HuggingFaceTB_SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# CORRECT prompt format for CodeT5
code = "def add(a, b):\n    return a + b"

prompt = f'''### Instruction:
Generate documentation for the following code:
{code}

### Response:'''
# prompt = f'{code}'
# prompt = f'Generate documentation for the code: {code}, and be concise and clear.'

inputs = tokenizer(prompt, return_tensors="pt", max_length=512)
outputs = model.generate(
    inputs.input_ids, 
    max_length=128,
    num_beams=4,  # Add beam search
    early_stopping=True
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {prompt}")
print(f"\nOutput: {result}")
# Extract documentation after '### Response:'
doc = result.split("### Response:")[-1].strip()
print(f"\nResponse: {doc}")