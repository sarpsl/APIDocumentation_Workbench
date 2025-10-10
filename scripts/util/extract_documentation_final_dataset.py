import json

with open("datasets/final/final_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# extract "documentation" field from each item as a list
documentation_list = [item["documentation"] for item in data]

print(documentation_list)
