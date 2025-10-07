import os
import json
from glob import glob

data_dir = "data"
for json_path in glob(os.path.join(data_dir, "*.json")):
    jsonl_path = json_path.replace(".json", ".jsonl")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Converted {json_path} -> {jsonl_path}")