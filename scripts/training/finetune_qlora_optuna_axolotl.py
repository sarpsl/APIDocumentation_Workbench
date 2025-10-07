import os
import json
import subprocess

# Paths to data
train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/train_cleaned_declarative.json'))
val_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/validation_cleaned_declarative.json'))
test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/test_cleaned_declarative.json'))

# Model name (Hugging Face Transformers compatible)
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Axolotl expects JSONL or similar format. This function can export your data if needed.
def export_for_axolotl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(output_path, 'w', encoding='utf-8') as out:
        for item in data:
            code = item.get('code', None)
            doc = item.get('doc', None)
            if code is not None and doc is not None:
                code_len = len(code)
                doc_len = len(doc)
                if 15 <= code_len <= 220 and 10 <= doc_len <= 120:
                    # Axolotl expects {"input": ..., "output": ...} or similar, depending on config
                    out.write(json.dumps({"input": code, "output": doc}) + "\n")

# Export datasets for Axolotl
# axolotl_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/axolotl_train.jsonl'))
# axolotl_val = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/axolotl_val.jsonl'))
# export_for_axolotl(train_path, axolotl_train)
# export_for_axolotl(val_path, axolotl_val)

if __name__ == "__main__":
    # Path to your Axolotl config YAML
    axolotl_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/axolotl_config.yaml'))
    # You must create this YAML file according to Axolotl documentation.
    # It should reference axolotl_train.jsonl and axolotl_val.jsonl as data sources.

    print("Launching Axolotl finetuning...")
    try:
        result = subprocess.run([
            "axolotl",
            axolotl_config,
        ], check=True)
        print("Axolotl finetuning completed.")
    except FileNotFoundError:
        print("Axolotl CLI not found. Please install axolotl and ensure it is in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Axolotl finetuning failed: {e}")

    print("\nNext steps:")
    print("- Check the Axolotl logs and output directory for your finetuned model.")
    print("- Adjust the YAML config for QLoRA, batch size, epochs, etc. as needed.")
    print("- See https://github.com/OpenAccess-AI-Collective/axolotl for more details.")
