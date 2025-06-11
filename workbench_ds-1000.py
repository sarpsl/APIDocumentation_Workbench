import urllib.request
import gzip
import json
import io

url = "https://raw.githubusercontent.com/xlang-ai/DS-1000/refs/heads/main/data/ds1000.jsonl.gz"

# Download the gzipped file into memory
with urllib.request.urlopen(url) as response:
    with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as gz:
        ds1000 = [json.loads(line) for line in gz]

print(f"Loaded {len(ds1000)} items.")
print(ds1000[0].keys())