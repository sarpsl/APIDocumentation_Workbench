from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load CodeT5
tokenizer = AutoTokenizer.from_pretrained("models/Salesforce_codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("models/Salesforce_codet5-small")

# CORRECT prompt format for CodeT5
code = "def add(a, b):\n    return a + b"
prompt = f'''### Instruction:
Generate documentation for the following code:
{code}

### Response:'''

inputs = tokenizer(prompt, return_tensors="pt", max_length=512)
outputs = model.generate(
    inputs.input_ids, 
    max_length=128,
    num_beams=4,  # Add beam search
    early_stopping=True
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {prompt}")
print(f"Output: {result}")