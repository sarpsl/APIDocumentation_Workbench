from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json

# model_name = "models/HuggingFaceTB_SmolLM2-1.7B-Instruct"
# model_name = "models/HuggingFaceTB_SmolLM2-360M-Instruct"
# model_name = "models/HuggingFaceTB_SmolLM2-360M-Instruct-Jarvislabs"
# model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# model_name = "models/Qwen_Qwen2.5-Coder-0.5B-Instruct"
model_name = "models/Qwen_Qwen2.5-Coder-0.5B-Instruct-Jarvislabs"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_documnetation(code, expected_doc=""):
    # prompt = f'''### Instruction:
    # Generate documentation for the following code:
    # {code}

    # ### Response:'''
    # prompt = f'{code}'
    # prompt = f'Generate documentation for the code: {code}, and be concise and clear.'
    prompt = f'Generate documentation for the following code:\n{code}\n\nDocumentation:\n'

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=256,
        num_beams=4,  # Add beam search
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nCode: {code}")
    # print(f"\nOutput: {result}")
    # Extract documentation after '### Response:'
    # doc = result.split("### Response:")[-1].strip()
    # print(f"\nExpected Documentation: {expected_doc}")
    doc = result.split("Documentation")[-1].strip()
    print(f"\nResponse: {doc}")

with open("data/test_cleaned_declarative-5.json") as f:
# with open("data/synthetic_test_data.json") as f:
    raw_data = json.load(f)

for item in raw_data:
    code = item['code']
    # expected_doc = item['doc']
    generate_documnetation(code)
