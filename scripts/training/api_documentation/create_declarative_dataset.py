import json

# File paths
# text_file_path = 'dataset-no-problems-documentation-declarative.txt'
text_file_path = 'dataset-no-problems-documentation-declarative-update1.txt'
input_json_path = 'dataset-no-problems-imperative.json'
output_json_path = 'dataset-no-problems-declarative.json'

# Step 1: Read lines from the text file
with open(text_file_path, 'r') as f:
    new_documentation_lines = [line.strip() for line in f.readlines()]

# Step 2: Load the JSON dataset
with open(input_json_path, 'r') as f:
    dataset = json.load(f)

# Step 3: Replace the 'documentation' field for each item
if len(new_documentation_lines) != len(dataset):
    raise ValueError("Mismatch between documentation lines and dataset entries")

for i, item in enumerate(dataset):
    item['documentation'] = new_documentation_lines[i]

# Step 4: Write the updated dataset to a new JSON file
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Updated dataset saved to {output_json_path}")
