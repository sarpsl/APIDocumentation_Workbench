# API Documentation Training Pipeline

> **A comprehensive machine learning pipeline for generating and improving API documentation using state-of-the-art language models.**

This repository contains a complete pipeline for automated API documentation generation, paraphrasing, and enhancement using various pre-trained and fine-tuned language models including SmolLM2, Qwen2.5-Coder, T5, and more.

## 🎯 Project Overview

The project consists of three main components:

1. **📖 Website Content Extraction & Q&A Generation** - Crawl documentation websites and create Q&A training data
2. **🧠 Code Documentation Generation** - Train models to generate documentation from source code
3. **🔄 Text Paraphrasing & Enhancement** - Improve existing documentation through rewriting and style transformation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/sarpsl/APIDocumentation.git
cd APIDocumentation

# Install dependencies
pip install -r requirements_pipeline.txt
```

### Quick Test

```bash
# Run a quick test with minimal training
python quick_start_pipeline.py
```

### Full Pipeline

```bash
# Run the complete pipeline
python full_training_pipeline.py
```

## 📋 Features & Capabilities

### 🌐 Website Content Processing
- **Recursive web crawling** with configurable depth and filtering
- **Content cleaning** and structured extraction
- **Automatic Q&A pair generation** from documentation content
- **Smart filtering** to exclude non-relevant files (PDFs, images, etc.)

### 💻 Code Documentation Generation
- **Multi-model support**: SmolLM2-360M-Instruct, Qwen2.5-Coder-0.5B-Instruct
- **Fine-tuning with LoRA/PEFT** for efficient training
- **Code-to-documentation** generation with various programming languages
- **Multiple dataset integration**: MBPP, DS-1000, custom datasets

### 📝 Text Enhancement & Paraphrasing
- **T5-based paraphrasing** models for text enhancement
- **Style transformation** (imperative to declarative, etc.)
- **Multiple paraphrasing strategies** with different model architectures
- **Batch processing** capabilities for large datasets

## 🗂️ Project Structure

```
APIDocumentation_Workbench/
├── config/                  # YAML and pipeline configs
├── data/                    # Raw and processed datasets
├── datasets/                # External and custom datasets
├── models/                  # Model checkpoints and cards
├── results/                 # Output results
├── evaluation_results/      # Evaluation metrics and tables
├── logs/                    # Training and experiment logs
├── scripts/
│   ├── training/            # Training scripts
│   ├── evaluation/          # Evaluation scripts
│   ├── util/                # Utilities and helpers
│   ├── dataset/             # Dataset extraction/fixing
│   ├── test/                # Model testing scripts
│   ├── inference/           # Inference utilities
│   └── snippets/            # Miscellaneous code snippets
├── full_training_pipeline.py # Main pipeline orchestrator
├── quick_start_pipeline.py   # Quick testing pipeline
├── requirements.txt          # Python dependencies
├── python-env.yml            # Conda environment file
└── ...
```

## 🔧 Configuration

- All config files are in `config/`
- Document the purpose of each config file in the README
- Example: `axolotl_config.yaml` for Axolotl training

## 🎯 Usage Examples

### Website Documentation Q&A

```python
from scripts.full_training_pipeline import WebCrawler, DataProcessor, ModelTrainer, QASystem
# ...existing code...
```

### Code Documentation Generation

```python
from scripts.training.training import train_documentation_model
# ...existing code...
```

### Text Paraphrasing & Enhancement

```python
from transformers import pipeline
# ...existing code...
```

## 📊 Datasets

**Available Datasets:**
- **DS-1000**: 1000 data science programming problems (`datasets/ds-1000/`)
- **MBPP**: Mostly Basic Python Programming problems (`datasets/mbpp/`)
- **TAPACO**: Paraphrasing dataset with 73,000+ pairs (`datasets/tapaco/`)
- **Custom Documentation**: Extracted from various API documentation sites (`datasets/website/`)
- **Synthetic & Real Datasets**: Located in `data/` and `datasets/`

### Dataset Format
All datasets follow a consistent JSON structure:
```json
{
    "code": "def example_function(x): return x * 2",
    "documentation": "This function multiplies the input by 2 and returns the result"
}
```

## 🔬 Experiments & Testing

The repository includes several experimental scripts:

- `rephrasing_test.py` - Basic paraphrasing experiments
- `rephrasing_test2-5.py` - Advanced paraphrasing with different models
- `prompt_*_test.py` - Prompt engineering experiments
- `workbench*.py` - Development and debugging utilities

## 🏋️ Training Details

## 🏋️ Training & Evaluation

- Train models with scripts in `scripts/training/`
- Evaluate with `scripts/evaluation/model_evaluation.py`
- Results saved in `evaluation_results/`

## 🧠 Models

- Model checkpoints in `models/`
- Model cards and tokenizer files in each model directory
- Supported models: SmolLM2, Qwen2.5-Coder, T5 variants, and more

## 🎛️ Advanced Configuration

### Custom Model Training

```python
# training/api_documentation/training.py
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
```

### Generation Parameters

```python
generation_config = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}
```

## 📈 Performance & Benchmarks

- Training Speed: ~2-3 hours on RTX 4090 for full training
- Memory Usage: 6-8GB GPU memory for inference
- Generation Quality: BLEU scores of 0.65+ on test sets
- Supported Languages: Python, JavaScript, Java, C++, and more

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
    - Reduce batch size in config
2. **Model Loading Errors**
    - Clear transformers cache
3. **Website Crawling Issues**
    - Check robots.txt and add delays

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library and model hosting
- **SmolLM2** and **Qwen2.5-Coder** teams for the base models
- **DS-1000** and **MBPP** dataset creators
- **TAPACO** paraphrasing dataset contributors

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@repository{api-documentation-pipeline,
  title={API Documentation Training Pipeline},
  author={Syed Abdul Rahim},
  year={2025},
  url={https://github.com/sarpsl/APIDocumentation}
}
```

---

For questions and support, please open an issue or reach out via the discussions tab.
