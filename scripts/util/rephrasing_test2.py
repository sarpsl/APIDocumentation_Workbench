from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define the input sentence
input_sentence = "Write a function that takes a string and returns it in uppercase."

# Format the input for the T5 model
input_text = f"rephrase: {input_sentence}"

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512)

# Generate the paraphrase
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the output
paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result
print(f"Original: {input_sentence}")
print(f"Paraphrase: {paraphrase}")
