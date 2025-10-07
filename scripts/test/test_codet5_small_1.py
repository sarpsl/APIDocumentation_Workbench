from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load CodeT5
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# Test single generation
code = "def add(a, b):\n    return a + b"
prompt = f"Generate documentation: {code}"

inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs.input_ids, max_length=128)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {prompt}")
print(f"Output: {result}")