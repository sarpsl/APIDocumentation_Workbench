# take a json file as input, filter to top N longest code+doc pairs, and output a new json file
import os
import json

# file_path = "data/train_cleaned_declarative.json"
file_path = "data/test_cleaned_declarative.json"

def filter_top_n(input_path, output_path, n=500):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {input_path}")
    
    # filter records basing on
    # 15 <= code_len <= 220 and 10 <= doc_len <= 120
    # until we have n records
    # save to output_path
    filtered = []
    for item in data:
        code = item.get('code', None)
        documentation = item.get('doc', None)
        if code is not None and documentation is not None:
            code_len = len(code)
            doc_len = len(documentation)
            if 15 <= code_len <= 220 and 10 <= doc_len <= 120:
                filtered.append(item)
                if len(filtered) >= n:
                    break
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"Filtered {len(filtered)} records to {output_path}")

if __name__ == "__main__":
    records_count = 100
    output_path = file_path.replace(".json", "-" + str(records_count) + ".json")
    filter_top_n(file_path, output_path, n=records_count)
    print(f"Filtered data saved to {output_path}")
