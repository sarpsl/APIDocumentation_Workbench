from transformers import pipeline

# Initialize the pipeline for text generation
text_transformer = pipeline(
    task="text-generation",
    # model="meta-llama/Llama-2-7b-hf",  # or another Llama model you have access to
    model="stabilityai/stablelm-3b-4e1t",  # You can use another StableLM variant if desired
    device=0  # Remove or set to -1 if you don't have a GPU
)

# Prompt for rephrasing
prompt = "Transform imperative to declarative: Write a function to split a string into parts"

result = text_transformer(prompt, max_new_tokens=50, num_return_sequences=1)
print("\n")
print(result)
print("\n")

# Extract only the part after 'Declarative:'
# if "Declarative:" in result:
#     reply = result.split("\n")[2].strip()
#     # Optionally, get only the first sentence
#     # reply = reply.split(".")[0].strip() + "."
# else:
#     reply = result

print(result['generated_text'].split("\n"))
