from transformers import pipeline

# model_name = "BEE-spoke-data/smol_llama-81M-tied"
# model_name = "HuggingFaceTB/SmolLM-360M-Instruct-WebGPU"
# model_name = "HuggingFaceTB/SmolLM2-360M"
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
pipe = pipeline("text-generation", model=model_name)

def get_response(prompt: str) -> str:
    response = pipe(prompt) #, max_new_tokens=60, do_sample=False)
    return response[0]['generated_text']

if __name__ == "__main__":
    print("Enter your prompt (type 'exit' to quit):")
    while True:
        prompt = input(">>> ")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        result = get_response(prompt)
        print("Result:", result)

