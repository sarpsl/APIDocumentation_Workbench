# Full Training Pipeline: Website to Q&A Model

This pipeline crawls a website, processes the content, and trains a custom Q&A model.

## 🚀 Quick Start


1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run quick test (5 pages, minimal training):**
```bash
python quick_start_pipeline.py
```

3. **Run full pipeline:**
```bash
python full_training_pipeline.py
```

## 📋 Pipeline Steps

### 1. Web Crawling
- Recursively crawls website pages
- Filters out non-HTML content (PDFs, images, etc.)
- Extracts and cleans text content
- Saves structured data

### 2. Data Processing
- Splits content into meaningful chunks
- Generates Q&A pairs automatically
- Creates various question types:
  - "What does the documentation say about...?"
  - "How can I...?"
  - "Can you explain...?"

### 3. Model Training
- Fine-tunes SmolLM2-360M-Instruct model
- Uses instruction tuning format
- Supports GPU acceleration
- Saves checkpoints during training

### 4. Interactive Q&A
- Loads the trained model
- Provides interactive chat interface
- Answers questions about the crawled content

## ⚙️ Configuration

Edit `pipeline_config.json` to customize:

- **Website URL** and crawling parameters
- **Training hyperparameters**
- **Model generation settings**
- **Output file paths**

## 📁 File Structure


```
website_crawler/
├── full_training_pipeline.py      # Main pipeline for crawling, Q&A generation, and training
├── quick_start_pipeline.py        # Quick test version (minimal crawl and training)
├── pipeline_config.json           # Configuration for crawling, training, and generation
├── requirements_pipeline.txt      # Python dependencies for this pipeline
├── datasets/
│   └── website/
│       ├── crawled_pages.json     # Raw crawled website data
│       └── qa_pairs.json          # Generated Q&A pairs from crawled data
└── README_PIPELINE.md             # This documentation file
```

## 🛠️ Customization

### Different Website
Change the `start_url` in config or directly in the script:
```python
WEBSITE_URL = "https://your-website.com"
```

### Different Model
Change the model in config:
```json
"model_name": "microsoft/DialoGPT-medium"
```

### Custom Question Generation
Modify the `_generate_questions_for_chunk()` method in `DataProcessor` class.

## 🎯 Usage Examples

### Basic Usage
```python
from full_training_pipeline import WebCrawler, DataProcessor, ModelTrainer, QASystem

# Crawl website
crawler = WebCrawler(max_pages=20)
pages = crawler.crawl_website("https://example.com")

# Process data
processor = DataProcessor()
qa_pairs = processor.generate_qa_pairs(pages)

# Train model
trainer = ModelTrainer()
dataset = trainer.prepare_training_data(qa_pairs)
model_path = trainer.train_model(dataset)

# Use Q&A system
qa_system = QASystem(model_path)
answer = qa_system.ask_question("How do I configure the system?")
```

### Interactive Mode
```python
qa_system = QASystem("./fine_tuned_qa_model")
qa_system.interactive_chat()
```

## 📊 Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust based on your GPU memory
3. **Epochs**: Start with 1-2 epochs for testing
4. **Pages**: Limit crawling for initial tests

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Use CPU training (slower)

2. **No Content Crawled**
   - Check website accessibility
   - Verify URL format
   - Check robots.txt permissions

3. **Training Fails**
   - Ensure sufficient Q&A pairs (>50)
   - Check data format
   - Verify model loading

### Debug Mode
Add debug prints in the pipeline:
```python
print(f"Pages crawled: {len(pages)}")
print(f"Q&A pairs: {len(qa_pairs)}")
print(f"Training samples: {len(dataset)}")
```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
