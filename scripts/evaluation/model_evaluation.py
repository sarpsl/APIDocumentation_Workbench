"""
Comprehensive Model Evaluation Script for Table 15
Evaluates 11 models (LLMs and SLMs) for API documentation generation

Models to evaluate:
LLMs: Code Llama, Falcon, Gemma, StarCoder, CodeBERT, CodeT5, Flan-T5
SLMs: Qwen 2, TinyLlama, Gemma-2, Phi-2

Metrics: BLEU, METEOR, ROUGE-L, BERTScore

Author: Syed Abdul Rahim
Date: 2025
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import shutil
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM
)
from datasets import load_dataset, load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
import time
import psutil
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
import nltk
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for model evaluation"""
    
    # Dataset paths
    TEST_DATASET_PATH = "data/test_cleaned_declarative_limited.json"
    
    # Number of samples to evaluate (use subset for testing, full for final)
    NUM_SAMPLES = 100  # Change to 1000+ for final evaluation
    
    # Models to evaluate
    MODELS = {
        # LLMs
        # "CodeT5": {
        #     "model_name": "Salesforce/codet5-base",
        #     "type": "seq2seq",
        #     "size": "220M"
        # },
        # "CodeT5-small": {
        #     "model_name": "Salesforce/codet5-small",
        #     "type": "seq2seq",
        #     "size": "60M"
        # },
        # "CodeBERT": {
        #     "model_name": "microsoft/codebert-base",
        #     "type": "causal",  # Used for code understanding, adapted for generation
        #     "size": "125M"
        # },
        # "Flan-T5": {
        #     "model_name": "google/flan-t5-base",
        #     "type": "seq2seq",
        #     "size": "250M"
        # },
        # "StarCoder": {
        #     "model_name": "bigcode/starcoderbase-1b",
        #     "type": "causal",
        #     "size": "1B"
        # },
        # "CodeLlama": {
        #     "model_name": "codellama/CodeLlama-7b-hf",
        #     "type": "causal",
        #     "size": "7B",
        #     "load_in_8bit": True  # Use quantization for large models
        # },
        # # SLMs
        # "TinyLlama": {
        #     "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        #     "type": "causal",
        #     "size": "1.1B"
        # },
        # "Phi-2": {
        #     "model_name": "microsoft/phi-2",
        #     "type": "causal",
        #     "size": "2.7B"
        # },
        # "Qwen-0.5B": {
        #     "model_name": "Qwen/Qwen2-0.5B",
        #     "type": "causal",
        #     "size": "0.5B"
        # },
        # "Gemma-2B": {
        #     "model_name": "google/gemma-2b",
        #     "type": "causal",
        #     "size": "2B"
        # },
        # "Gemma-2-2B": {
        #     "model_name": "google/gemma-2-2b",
        #     "type": "causal",
        #     "size": "2B"
        # },
        # "SmolLM2-360M-Instruct": {
        #     "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        #     "type": "causal",
        #     "size": "360M"
        # },
        # "SmolLM2-360M-Instruct-Finetuned": {
        #     "model_name": "models/HuggingFaceTB_SmolLM2-360M-Instruct",
        #     "type": "causal",
        #     "size": "360M"
        # },
        # "Flan-T5-Finetuned": {
        #     "model_name": "models/google_flan-t5-base",
        #     "type": "seq2seq",
        #     "size": "250M"
        # },
        # "Flan-T5-Finetuned2": {
        #     "model_name": "models/google_flan-t5-base_0",
        #     "type": "seq2seq",
        #     "size": "250M"
        # },
        # "Qwen-0.5B-Finetuned2": {
        #     "model_name": "models/Qwen_Qwen2.5-Coder-0.5B-Instruct",
        #     "type": "causal",
        #     "size": "0.5B"
        # },
        # "CodeT5-small-Finetuned2": {
        #     "model_name": "models/Salesforce_codet5-small",
        #     "type": "seq2seq",
        #     "size": "220M"
        # },
        "SmolLM2-360M-Instruct-Finetuned-Jarvislabs": {
            "model_name": "models/HuggingFaceTB_SmolLM2-360M-Instruct-Jarvislabs",
            "type": "causal",
            "size": "360M"
        },
        "Qwen-0.5B-Instruct-Finetuned-Jarvislabs": {
            "model_name": "models/Qwen_Qwen2.5-Coder-0.5B-Instruct-Jarvislabs",
            "type": "causal",
            "size": "0.5B"
        }
    }
    
    # Prompt template
    PROMPT_TEMPLATE = """Generate documentation for the following Python code:

Code:
{code}

Documentation:"""
    
    # Generation parameters
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 128
    GENERATION_CONFIG = {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "num_beams": 1
    }
    
    # Output
    RESULTS_DIR = "evaluation_results"
    RESULTS_FILE = "model_evaluation_results.csv"

def clear_huggingface_cache():
    """Delete Hugging Face cache directory to free disk space."""
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    if os.path.exists(cache_dir):
        print(f"Clearing Hugging Face cache at {cache_dir}...")
        try:
            shutil.rmtree(cache_dir)
            print("✓ Cache cleared.")
        except Exception as e:
            print(f"✗ Failed to clear cache: {e}")
    else:
        print("No Hugging Face cache found to clear.")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(path, num_samples=None):
    """Load and prepare test dataset"""
    print(f"Loading test data from {path}...")
    
    # Try loading as JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to list of dicts if needed
        if isinstance(data, dict):
            data = [{"code": k, "doc": v} for k, v in data.items()]
    except:
        # Try loading as dataset
        data = load_from_disk(path)
        data = [{"code": item["code"], "documentation": item["doc"]} 
                for item in data]
    
    # Limit samples if specified
    if num_samples:
        data = data[:num_samples]
    
    print(f"Loaded {len(data)} samples")
    return data


# ============================================================================
# MODEL LOADING AND INFERENCE
# ============================================================================

class ModelEvaluator:
    """Handles model loading and inference"""
    
    def __init__(self, model_config, device="cuda"):
        self.config = model_config
        # self.device = device if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"\nLoading {self.config['model_name']}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.config['type'] == 'seq2seq':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config['model_name'],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                load_kwargs = {
                    'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32
                }
                if self.config.get('load_in_8bit', False):
                    load_kwargs['load_in_8bit'] = True
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config['model_name'],
                    **load_kwargs
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return False
    
    def generate_documentation(self, code, prompt_template):
        """Generate documentation for given code"""
        if self.model is None:
            return None
        
        # Format prompt
        prompt = prompt_template.format(code=code)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=Config.MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                if self.config['type'] == 'seq2seq':
                    outputs = self.model.generate(
                        **inputs,
                        # max_length=Config.MAX_OUTPUT_LENGTH,
                        **Config.GENERATION_CONFIG
                    )
                else:
                    # For causal models, continue from prompt
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **Config.GENERATION_CONFIG
                    )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract documentation part (after "Documentation:")
            if "Documentation:" in generated_text:
                doc = generated_text.split("Documentation:")[-1].strip()
            else:
                doc = generated_text.replace(prompt, "").strip()
            
            return doc
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def unload_model(self):
        """Free model from memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()


# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score"""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if not hyp_tokens:
            return 0.0
        
        return sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=self.smoothing
        )
    
    def calculate_meteor(self, reference, hypothesis):
        """Calculate METEOR score"""
        if not hypothesis.strip():
            return 0.0
        
        return meteor_score([reference.split()], hypothesis.split())
    
    def calculate_rouge(self, reference, hypothesis):
        """Calculate ROUGE-L score"""
        if not hypothesis.strip():
            return 0.0
        
        scores = self.rouge_scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure
    
    def calculate_bertscore(self, references, hypotheses):
        """Calculate BERTScore for batch"""
        if not hypotheses or all(not h.strip() for h in hypotheses):
            return 0.0, 0.0, 0.0
        
        P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def calculate_all_metrics(self, references, hypotheses):
        """Calculate all metrics"""
        bleu_scores = []
        meteor_scores = []
        rouge_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            bleu_scores.append(self.calculate_bleu(ref, hyp))
            meteor_scores.append(self.calculate_meteor(ref, hyp))
            rouge_scores.append(self.calculate_rouge(ref, hyp))
        
        # BERTScore (batch)
        bert_p, bert_r, bert_f1 = self.calculate_bertscore(references, hypotheses)
        
        return {
            'bleu': np.mean(bleu_scores),
            'meteor': np.mean(meteor_scores),
            'rouge_l': np.mean(rouge_scores),
            'bertscore_precision': bert_p,
            'bertscore_recall': bert_r,
            'bertscore_f1': bert_f1
        }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_model(model_name, model_config, test_data):
    """Evaluate single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Track time and memory
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    
    # Load model
    evaluator = ModelEvaluator(model_config, device="cpu")
    if not evaluator.load_model():
        return None
    
    # Generate documentation
    references = []
    hypotheses = []
    
    print(f"\nGenerating documentation for {len(test_data)} samples...")
    for item in tqdm(test_data):
        code = item['code']
        ref_doc = item['doc']
        
        gen_doc = evaluator.generate_documentation(code, Config.PROMPT_TEMPLATE)
        
        references.append(ref_doc)
        hypotheses.append(gen_doc if gen_doc else "")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(references, hypotheses)
    
    # Calculate resource usage
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_after = process.memory_info().rss / 1024 / 1024 / 1024
    memory_used = max(memory_after - memory_before, 0.1)  # Minimum 0.1GB
    
    # Estimate tokens per second
    total_tokens = sum(len(h.split()) for h in hypotheses)
    tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    # Unload model
    evaluator.unload_model()
	
	# Clear cache after unloading model
    #clear_huggingface_cache()
    
    # Compile results
    results = {
        'model': model_name,
        'type': 'LLM' if model_config['size'].endswith(('B', 'M')) and 
                float(model_config['size'][:-1]) >= 1 else 'SLM',
        'size': model_config['size'],
        'bleu': f"{metrics['bleu']:.3f}",
        'meteor': f"{metrics['meteor']:.3f}",
        'rouge_l': f"{metrics['rouge_l']:.3f}",
        'bertscore_f1': f"{metrics['bertscore_f1']:.3f}",
        'inference_time_sec': f"{elapsed_time:.1f}",
        'memory_gb': f"{memory_used:.1f}",
        'tokens_per_sec': f"{tokens_per_second:.1f}",
        'samples_evaluated': len(test_data)
    }
    
    print(f"\nResults for {model_name}:")
    print(f"  BLEU: {results['bleu']}")
    print(f"  METEOR: {results['meteor']}")
    print(f"  ROUGE-L: {results['rouge_l']}")
    print(f"  BERTScore F1: {results['bertscore_f1']}")
    print(f"  Time: {results['inference_time_sec']}s")
    print(f"  Memory: {results['memory_gb']} GB")
    
    return results


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("MODEL EVALUATION FOR TABLE 15")
    print("="*80)
    
    # Create results directory
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Load test data
    test_data = load_test_data(Config.TEST_DATASET_PATH, Config.NUM_SAMPLES)
    
    # Evaluate all models
    all_results = []
    
    for model_name, model_config in Config.MODELS.items():
        try:
            results = evaluate_model(model_name, model_config, test_data)
            if results:
                all_results.append(results)
                
                # Save intermediate results
                df = pd.DataFrame(all_results)
                df.to_csv(
                    os.path.join(Config.RESULTS_DIR, Config.RESULTS_FILE),
                    index=False
                )
                print(f"\n✓ Results saved to {Config.RESULTS_FILE}")
                
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            continue
    
    # Final results summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\nFinal Results Summary:")
        print(df.to_string(index=False))
        
        # Save final results
        final_path = os.path.join(Config.RESULTS_DIR, "final_" + Config.RESULTS_FILE)
        df.to_csv(final_path, index=False)
        
        # Also save as formatted table for thesis
        table_path = os.path.join(Config.RESULTS_DIR, "table_15_formatted.txt")
        with open(table_path, 'w') as f:
            f.write("Table 15: Evaluation Results and Analysis\n\n")
            f.write(df.to_string(index=False))
        
        print(f"\n✓ All results saved to {Config.RESULTS_DIR}/")
    else:
        print("\n✗ No models were successfully evaluated")


if __name__ == "__main__":
    main()