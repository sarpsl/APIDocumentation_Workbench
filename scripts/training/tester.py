
from transformers import pipeline
import evaluate, nltk
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from tqdm import tqdm
import time
import re
nltk.download('wordnet')
nltk.download('punkt')

class Tester:
    def __init__(self, model_name, model, tokenizer, task="text-generation", prompt="{code}", device=0):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.task = task
        self.prompt = prompt
        self.pipe = pipeline(
            self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.bleu_metric = evaluate.load("bleu")
        self.meteor_metric = evaluate.load("meteor")
        self.rouge_metric = evaluate.load("rouge")
        self.total_records = 0
        self.skipped_records = 0

    def load_examples(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        examples = [
            {"expected": item["doc"], "code": item["code"]}
            for item in data
            if "doc" in item and "code" in item
        ]
        self.total_records = len(examples)
        return examples

    def run(self, file_path, max_records=100):
        examples = self.load_examples(file_path)
        preds = []
        refs = []
        inference_times = []

        for idx, item in enumerate(tqdm(examples, desc="Evaluating examples"), 1):
            if (idx - self.skipped_records) > max_records:
                break

            code_snippet = item["code"]
            expected_doc = item.get("expected", "N/A")

            # skip item if the expected_doc existing in code_snippet
            if expected_doc in code_snippet:
                self.skipped_records += 1
                continue
            # Skip if expected_doc contains HTML or other markup
            if re.search(r"<[^>]+>", expected_doc) or "@" in expected_doc:
                self.skipped_records += 1
                continue

            prompt = self.prompt.format(code=code_snippet)
            
            start_time = time.time()
            result = self.pipe(
                prompt,
                max_new_tokens=150,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )[0]["generated_text"]
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            response = result.split("### Response:")[-1].strip()
            preds.append(response)
            refs.append([expected_doc])

            print(f"\n--- Example {idx} ---")
            print(f"Code:\n{code_snippet}\n")
            print(f"\nGenerated Documentation:\n{response}")
            print(f"\nExpected Documentation:\n{expected_doc}")
            print(f"\nInference time: {inference_time:.3f} seconds")
            print("-" * 50)

        self.evaluate(preds, refs, inference_times)

    def evaluate(self, preds, refs, inference_times):
        print("\n=== Evaluation ===")
        print(f"Model Name: {self.model_name}")
        print(f"Total Records: {self.total_records}")
        print(f"Skipped Records: {self.skipped_records}")
        # only evaluate if there are predictions
        if not preds or not refs:
            print("No predictions to evaluate.")
            return
        
        bleu_score = self.bleu_metric.compute(predictions=preds, references=refs)
        meteor_score = self.meteor_metric.compute(predictions=preds, references=[r[0] for r in refs])
        rouge_score = self.rouge_metric.compute(predictions=preds, references=[r[0] for r in refs])

        def tokenize_for_metrics(text):
            return nltk.word_tokenize(text.lower())

        all_preds_tokens = [tokenize_for_metrics(p) for p in preds]
        all_refs_tokens = [tokenize_for_metrics(r[0]) for r in refs]
        preds_flat = sum(all_preds_tokens, [])
        refs_flat = sum(all_refs_tokens, [])
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
        
        print(f"BLEU score: {bleu_score['bleu']}")
        print(f"METEOR score: {meteor_score['meteor']}")
        print(f"ROUGE score: {rouge_score}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average inference time per example: {avg_inference_time:.3f} seconds")
