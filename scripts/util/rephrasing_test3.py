from transformers import pipeline

# Initialize the pipeline for text-to-text generation
text_transformer = pipeline(task="text2text-generation", model="google-t5/t5-base")

# Transform an imperative sentence to declarative
result = text_transformer("transform imperative to declarative: Close the window.")
# print(result['generated_text'])  # Output: He asked to close the window.
print(result)
