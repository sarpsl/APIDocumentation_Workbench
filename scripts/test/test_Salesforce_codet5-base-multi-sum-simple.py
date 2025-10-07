from transformers import RobertaTokenizer, T5ForConditionalGeneration

MODEL_NAME = "Salesforce/codet5-base-multi-sum"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_doc(code, max_length=128):
    prompt = f"{code}"
    # prompt = f"### Instruction: Generate documentation for the following code.\n### Code:\n{code}\n### Documentation:"
    # prompt = f'### Please be concise. \
    #                 Code: {code} \
    #                 For the code above, write documentation on what it does, not explaining the code itself and without repeating prompt or code.###'
    # prompt = f'For the code below, generate documentation on what it does, and not explaining the code itself.\
    #                 Code: {code} \
    #                 Documentation (only the answer): '
    # prompt = (
    #     "For the code below, generate documentation on what it does, not explaining the code itself.\n"
    #     "Code: def add(a, b): return a + b\nDocumentation: Returns the sum of two numbers.\n"
    #     "Code: def is_even(n): return n % 2 == 0\nDocumentation: Checks if a number is even.\n"
    #     "Code: def greet(name): print(f'Hello, {name}!')\nDocumentation: Prints a greeting message with the given name.\n"
    #     "Code: def factorial(n): return 1 if n == 0 else n * factorial(n-1)\nDocumentation: Calculates the factorial of a number recursively.\n"
    #     f"Code: {code}\nDocumentation: "
    # )
    print("\nPrompt: " + prompt + "\n")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    eos_token_id = tokenizer.eos_token_id
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id
    )
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nOutput: => " + output + "\n")
    # Extract only the documentation for the last code block
    # Find the last 'Documentation:' and return the text after it, up to the next 'Code:' or end
    if 'Documentation:' in output:
        documentation = output.split('Documentation:')[-1]
        documentation = documentation.split('Code:')[0].strip()
        return documentation
    return output.strip()

print("\nGenerated Doc Output: " + generate_doc("def add(a, b): return a + b", max_length=1024))