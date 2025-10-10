#!/usr/bin/env python3
"""
Full Training Pipeline: Web Crawling → Data Processing → Model Training → Q&A System

This pipeline:
1. Crawls a website recursively
2. Cleans and processes the content
3. Generates Q&A pairs from the content
4. Fine-tunes a language model
5. Creates an interactive Q&A system
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import re
from typing import List, Dict, Tuple
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from tqdm import tqdm
import torch

class WebCrawler:
    """Crawls website content recursively"""
    
    def __init__(self, max_pages=100, delay=1):
        self.max_pages = max_pages
        self.delay = delay
    
    def crawl_website(self, start_url: str) -> List[Dict]:
        """Crawl website and return structured content"""
        visited = set()
        to_visit = [start_url]
        pages = []
        
        print(f"🕷️  Starting crawl from: {start_url}")
        print(f"📄 Maximum pages: {self.max_pages}")
        print("-" * 60)

        while to_visit and len(visited) < self.max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            
            progress = f"[{len(visited)}/{self.max_pages}]"
            print(f"{progress} Crawling: {url}")
            
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract title
                title = soup.find('title')
                title = title.get_text().strip() if title else f"Page {len(visited)}"
                
                # Remove script, style, and navigation elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                
                # Extract main content
                main_content = soup.get_text(separator="\n")
                cleaned_content = self._clean_text(main_content)
                
                if len(cleaned_content.strip()) > 100:  # Only save meaningful content
                    pages.append({
                        "url": url,
                        "title": title,
                        "content": cleaned_content,
                        "length": len(cleaned_content)
                    })
                
                # Find new links
                new_links = self._extract_links(soup, url, start_url, visited, to_visit)
                print(f"    ✓ Extracted {len(cleaned_content)} chars, found {new_links} new links")
                
                # Be polite - add delay
                if self.delay > 0:
                    import time
                    time.sleep(self.delay)
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        
        print(f"\n🎉 Crawling completed! {len(pages)} pages extracted")
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove very short lines (likely navigation/UI elements)
        lines = text.split('\n')
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        return '\n'.join(meaningful_lines)
    
    def _extract_links(self, soup, current_url, start_url, visited, to_visit):
        """Extract valid links from page"""
        new_links_count = 0
        
        for a in soup.find_all("a", href=True):
            link = urljoin(current_url, a["href"])
            parsed_link = urlparse(link)
            path = parsed_link.path.lower()
            
            # Skip non-HTML files
            skip_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                             '.bmp', '.svg', '.mp3', '.mp4', '.avi', '.mov', '.css', '.js')
            
            if any(path.endswith(ext) for ext in skip_extensions):
                continue
            
            # Only crawl same domain
            if (urlparse(link).netloc == urlparse(start_url).netloc and 
                link not in visited and link not in to_visit):
                to_visit.append(link)
                new_links_count += 1
        
        return new_links_count


class DataProcessor:
    """Processes crawled content into training data"""
    
    def generate_qa_pairs(self, pages: List[Dict]) -> List[Dict]:
        """Generate Q&A pairs from crawled content"""
        qa_pairs = []
        
        print("🔄 Generating Q&A pairs from content...")
        
        for page in tqdm(pages, desc="Processing pages"):
            content = page['content']
            title = page['title']
            
            # Split content into chunks
            chunks = self._split_into_chunks(content, max_length=500)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                # Generate different types of questions
                qa_pairs.extend(self._generate_questions_for_chunk(chunk, title, i))
        
        print(f"📝 Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def _split_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into meaningful chunks"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_questions_for_chunk(self, chunk: str, title: str, chunk_idx: int) -> List[Dict]:
        """Generate various question types for a text chunk"""
        qa_pairs = []
        
        # Simple question patterns
        patterns = [
            f"What does the documentation say about {title.lower()}?",
            f"How can I {self._extract_action_from_text(chunk)}?",
            f"What is explained in {title}?",
            f"Can you explain {self._extract_concept_from_text(chunk)}?",
        ]
        
        for pattern in patterns:
            if pattern and len(pattern) > 10:
                qa_pairs.append({
                    "instruction": pattern,
                    "response": chunk,
                    "source_title": title,
                    "chunk_id": chunk_idx
                })
        
        return qa_pairs
    
    def _extract_action_from_text(self, text: str) -> str:
        """Extract actionable phrases from text"""
        # Look for verb phrases
        action_words = re.findall(r'\b(?:configure|setup|install|create|manage|use|access|enable|disable|run|execute|start|stop)\s+\w+', text.lower())
        return action_words[0] if action_words else "use this feature"
    
    def _extract_concept_from_text(self, text: str) -> str:
        """Extract key concepts from text"""
        # Look for important terms (capitalized words, technical terms)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return concepts[0] if concepts else "this concept"


class ModelTrainer:
    """Handles model training and fine-tuning"""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Using device: {self.device}")
    
    def prepare_training_data(self, qa_pairs: List[Dict]) -> Dataset:
        """Prepare data for training"""
        print("📊 Preparing training dataset...")
        
        # Format data for instruction tuning
        formatted_data = []
        for pair in qa_pairs:
            formatted_text = f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['response']}"
            formatted_data.append({"text": formatted_text})
        
        # Convert to Dataset
        df = pd.DataFrame(formatted_data)
        dataset = Dataset.from_pandas(df)
        
        print(f"📈 Training dataset size: {len(dataset)} examples")
        return dataset
    
    def train_model(self, dataset: Dataset, output_dir: str = "./fine_tuned_model"):
        """Fine-tune the model"""
        print("🏋️  Starting model training...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Split dataset for training and validation
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_test_split["train"],
            eval_dataset=train_test_split["test"],
        )
        
        # Train
        print("🚀 Training started...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        return output_dir


class QASystem:
    """Interactive Q&A system using the trained model"""
    
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("💬 Loading Q&A system...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Q&A system ready!")
    
    def ask_question(self, question: str) -> str:
        """Get answer for a question"""
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in response:
            answer = response.split("### Response:")[-1].strip()
        else:
            answer = response.split(prompt)[-1].strip() if prompt in response else response
        
        return answer
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "="*60)
        print("🤖 Interactive Q&A System")
        print("Ask questions about your documentation!")
        print("Type 'exit' or 'quit' to stop")
        print("="*60)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...")
            answer = self.ask_question(question)
            print(f"\n💡 Answer: {answer}")


def main():
    """Run the complete pipeline"""
    print("🚀 Starting Full Training Pipeline")
    print("="*60)
    
    # Configuration
    WEBSITE_URL = "https://bookshelf.erwin.com/bookshelf/15.0DIBookshelf/Content/Home.htm"
    MAX_PAGES = 100  # Adjust based on your needs
    OUTPUT_DIR = "./fine_tuned_qa_model"
    
    # Step 1: Crawl Website
    print("\n📡 STEP 1: Crawling Website")
    crawler = WebCrawler(max_pages=MAX_PAGES, delay=1)
    pages = crawler.crawl_website(WEBSITE_URL)
    
    if not pages:
        print("❌ No content crawled. Exiting.")
        return
    
    # Save crawled data
    with open("datasets/website/crawled_pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    # Step 2: Process Data
    print("\n🔧 STEP 2: Processing Data")
    processor = DataProcessor()
    qa_pairs = processor.generate_qa_pairs(pages)
    
    # Save Q&A pairs
    with open("datasets/website/qa_pairs.json", "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    # Step 3: Train Model
    print("\n🏋️  STEP 3: Training Model")
    trainer = ModelTrainer()
    dataset = trainer.prepare_training_data(qa_pairs)
    model_path = trainer.train_model(dataset, OUTPUT_DIR)
    
    # Step 4: Interactive Q&A
    print("\n💬 STEP 4: Starting Q&A System")
    qa_system = QASystem(model_path)
    qa_system.interactive_chat()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("datasets/website", exist_ok=True)
    
    # Run the pipeline
    main()
