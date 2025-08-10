#!/usr/bin/env python3
"""
Quick Start: Minimal Pipeline for Testing

This is a simplified version for quick testing with fewer pages and epochs.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from full_training_pipeline import WebCrawler, DataProcessor, ModelTrainer, QASystem
import json

def quick_start():
    """Run a minimal pipeline for testing"""
    print("🚀 Quick Start Pipeline")
    print("="*40)
    
    # Configuration for quick test
    WEBSITE_URL = "https://bookshelf.erwin.com/bookshelf/15.0DIBookshelf/Content/Home.htm"
    MAX_PAGES = 5  # Small number for testing
    OUTPUT_DIR = "./quick_test_model"
    
    try:
        # Step 1: Crawl (few pages)
        print("\n📡 Crawling 5 pages...")
        crawler = WebCrawler(max_pages=MAX_PAGES, delay=0.5)
        pages = crawler.crawl_website(WEBSITE_URL)
        
        if not pages:
            print("❌ No content crawled.")
            return False
        
        print(f"✅ Crawled {len(pages)} pages")
        
        # Step 2: Generate Q&A pairs
        print("\n🔧 Generating Q&A pairs...")
        processor = DataProcessor()
        qa_pairs = processor.generate_qa_pairs(pages)
        
        if len(qa_pairs) < 10:
            print("❌ Too few Q&A pairs generated.")
            return False
        
        print(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        # Step 3: Quick training (1 epoch)
        print("\n🏋️  Quick training (1 epoch)...")
        trainer = ModelTrainer()
        dataset = trainer.prepare_training_data(qa_pairs)
        
        # Override training args for quick test
        trainer.model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
        
        # Train with minimal settings
        model_path = trainer.train_model(dataset, OUTPUT_DIR)
        
        print("✅ Training completed!")
        
        # Step 4: Test Q&A
        print("\n💬 Testing Q&A system...")
        qa_system = QASystem(model_path)
        
        # Test with a sample question
        test_question = "What is this documentation about?"
        answer = qa_system.ask_question(test_question)
        
        print(f"\n❓ Test Question: {test_question}")
        print(f"💡 Answer: {answer}")
        
        print("\n🎉 Quick start completed successfully!")
        
        # Ask user if they want to continue with interactive mode
        user_input = input("\nDo you want to start interactive Q&A? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            qa_system.interactive_chat()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        return False

if __name__ == "__main__":
    # Create directories
    os.makedirs("datasets/website", exist_ok=True)
    
    success = quick_start()
    
    if success:
        print("\n✨ Pipeline completed successfully!")
        print("You can now run the full pipeline with more pages and epochs.")
    else:
        print("\n💥 Pipeline failed. Check the errors above.")
