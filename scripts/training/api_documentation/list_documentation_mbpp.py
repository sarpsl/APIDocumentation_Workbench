import json

with open('dataset-no-problems-imperative.json', 'r') as f:
    ds = json.load(f)

with open('dataset-no-problems-documentation-imperative.txt', 'w') as f:
    f.write('\n'.join(item['documentation'] for item in ds))

