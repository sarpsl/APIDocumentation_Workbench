import urllib.request
import json
import gzip
import io


def get_mbpp_subset():
    url = "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/mbpp/sanitized-mbpp.json"
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
    return [
        {"documentation": item["prompt"], "code": item["code"]}
        for item in data
        if "prompt" in item and "code" in item
    ]

def get_ds_1000_subset():
    url = "https://raw.githubusercontent.com/xlang-ai/DS-1000/refs/heads/main/data/ds1000.jsonl.gz"
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as gz:
            data = [json.loads(line) for line in gz]
    return [
        {"documentation": item["prompt"], "code": item["code_context"]}
        for item in data
        if "prompt" in item and "code_context" in item
    ]

# def get_codesearchnet_subset():
#     url = "https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/data/python/final/jsonl/test/python_test_0.jsonl"
#     with urllib.request.urlopen(url) as response:
#         data = [json.loads(line) for line in response]
#     return [
#         {"documentation": item["docstring"], "code": item["code"]}
#         for item in data
#         if "docstring" in item and "code" in item
#     ]

final_dataset = get_mbpp_subset() + get_ds_1000_subset() # + get_codesearchnet_subset()

# ...existing code...

final_dataset = get_mbpp_subset() + get_ds_1000_subset() # + get_codesearchnet_subset()

print(f"Dataset has {len(final_dataset)} items.")
print(final_dataset[0])

# Save final_dataset to a JSON file
with open("datasets/final/final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)
