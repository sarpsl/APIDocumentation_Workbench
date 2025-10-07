from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate, nltk
from sklearn.metrics import precision_score, recall_score, f1_score
import json
nltk.download('wordnet')
nltk.download('punkt')

# Load BLEU and METEOR metrics
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")
rogue_score = evaluate.load("rouge")

# === Load fine-tuned model ===
# model_path = "./smollm2-docgen"
# model_path = "./qwen-coder-docgen"
model_path = "./smollm2-1.7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # use -1 for CPU
)

# === Input: Array of code snippets with expected documentation ===
# Load examples from JSON file, renaming 'doc' to 'expected' and keeping only 'expected' and 'code'
# file_path = "data/test_cleaned_declarative-10.json"
file_path = "data/test_cleaned_declarative-7.json"
with open(file_path, "r") as f:
    data = json.load(f)

examples = [
    {"expected": item["doc"], "code": item["code"]}
    for item in data
    if "doc" in item and "code" in item
]

# Generate documentation
preds = []
refs = []
inference_times = []

# === Inference ===
for idx, item in enumerate(examples, 1):
    code_snippet = item["code"]
    expected_doc = item.get("expected", "N/A")

    prompt = f"""### Instruction:
Please be concise. Generate documentation without explaining for the following code:
{code_snippet}

### Response:
"""

    import time
    start_time = time.time()
    result = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    # Extract only the response portion (after '### Response:')
    response = result.split("### Response:")[-1].strip()

    preds.append(response)
    refs.append([expected_doc])  # note: refs should be list of list

    print(f"\n--- Example {idx} ---")
    print(f"Code:\n{code_snippet}\n")
    print(f"\nGenerated Documentation:\n{response}")
    print(f"\nExpected Documentation:\n{expected_doc}")
    print(f"\nInference time: {inference_time:.3f} seconds")
    print("-" * 50)

# Evaluate
bleu_score = bleu_metric.compute(predictions=preds, references=refs)
meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
rogue_score = evaluate.load("rouge").compute(predictions=preds, references=[r[0] for r in refs])

# Compute F1, Precision, Recall at token level
def tokenize_for_metrics(text):
    return nltk.word_tokenize(text.lower())

all_preds_tokens = [tokenize_for_metrics(p) for p in preds]
all_refs_tokens = [tokenize_for_metrics(r[0]) for r in refs]

# Flatten for micro-averaged metrics
preds_flat = sum(all_preds_tokens, [])
refs_flat = sum(all_refs_tokens, [])

# Create sets for unique tokens in both preds and refs
all_tokens = list(set(preds_flat + refs_flat))
token2idx = {tok: i for i, tok in enumerate(all_tokens)}

def to_binary_vector(tokens, token2idx):
    vec = [0] * len(token2idx)
    for t in tokens:
        if t in token2idx:
            vec[token2idx[t]] = 1
    return vec

y_true = to_binary_vector(refs_flat, token2idx)
y_pred = to_binary_vector(preds_flat, token2idx)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

print("\n=== Evaluation ===")
print("Model Path: ", model_path)
print("BLEU score:", bleu_score["bleu"])
print("METEOR score:", meteor_score["meteor"])
print("ROUGE score:", rogue_score)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Average inference time per example: {avg_inference_time:.3f} seconds")
