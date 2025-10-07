from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import evaluate, nltk
nltk.download('wordnet')
nltk.download('punkt')

# Load BLEU and METEOR metrics
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

# === Load fine-tuned model ===
# model_path = "./smollm2-docgen"
model_path = "./qwen-coder-docgen"
# model_path = "./smollm2-1.7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # use -1 for CPU
)

# === Input: Array of code snippets with expected documentation ===
examples = [
    {
	"expected": "This function returns the multiplication of two numbers",
	"code": "def multiply(a, b): return a * b"
},
{
	"expected": "This function checks if all elements in an array are unique",
	"code": "def test_duplicate(arraynums):\n    nums_set = set(arraynums)    \n    return len(arraynums) != len(nums_set)     "
}
,
{
	"expected": "Computes the factorial of a number using recursion.",
	"code": """def factorial(n):
     if n == 0:
         return 1
     else:
         return n * factorial(n - 1)"""
},
{
	"expected": "Classifies a score into a letter grade based on standard grading criteria.",
	"code": """def classify_grade(score):
     if score >= 90:
         return "A"
     elif score >= 80:
         return "B"
     elif score >= 70:
         return "C"
     else:
         return "F"
         """
},
{
	"expected": "Computes the nth Fibonacci number using recursion with memoization to optimize performance by caching results.",
	"code": """def fibonacci(n, memo={}):
     if n in memo:
         return memo[n]
     if n <= 1:
         return n
     memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
     return memo[n]
     """
},
{
	"expected": "Validates whether a given string is a valid email address using a regular expression.",
	"code": """import re
 def is_valid_email(email):
     pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
     return re.match(pattern, email) is not None
     """
},{
	"expected": "Sorts a dictionary by its values in ascending or descending order.",
	"code": """def sort_dict_by_value(d, reverse=False):
     return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))
     """
},{
	"expected": "Recursively flattens a nested list into a single-level list.",
	"code": """def flatten_list(nested_list):
     flat_list = []
     for item in nested_list:
         if isinstance(item, list):
             flat_list.extend(flatten_list(item))
         else:
             flat_list.append(item)
     return flat_list

 """
},{
	"expected": "Reads a file and returns the number of words contained in it.",
	"code": """def count_words_in_file(filename):
     with open(filename, 'r') as file:
         text = file.read()
     words = text.split()
     return len(words)
     """
},{
	"expected": "Groups a list of items by their category field and returns a dictionary mapping each category to a list of item names.",
	"code": """from collections import defaultdict
 def group_items_by_category(items):
     grouped = defaultdict(list)
     for item in items:
         grouped[item['category']].append(item['name'])
     return dict(grouped)
 """
},{
	"expected": "Removes duplicates from a list while preserving the original order of elements.",
	"code": """def remove_duplicates(seq):
     seen = set()
     result = []
     for item in seq:
         if item not in seen:
             seen.add(item)
             result.append(item)
     return result
 """
}
]

# Generate documentation
preds = []
refs = []

# === Inference ===
for idx, item in enumerate(examples, 1):
    code_snippet = item["code"]
    expected_doc = item.get("expected", "N/A")

    prompt = f"""### Instruction:
Generate documentation for the following code:
{code_snippet}

### Response:
"""

    result = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]

    # Extract only the response portion (after '### Response:')
    response = result.split("### Response:")[-1].strip()

    preds.append(response)
    refs.append([expected_doc])  # note: refs should be list of list

    print(f"\n--- Example {idx} ---")
    print(f"Code:\n{code_snippet}\n")
    print(f"Generated Documentation:\n{response}")
    print(f"Expected Documentation:\n{expected_doc}")
    print("-" * 50)

# Evaluate
bleu_score = bleu_metric.compute(predictions=preds, references=refs)
meteor_score = meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])

print("\n=== Evaluation ===")
print("BLEU score:", bleu_score["bleu"])
print("METEOR score:", meteor_score["meteor"])
