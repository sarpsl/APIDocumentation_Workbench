from transformers import RobertaTokenizer, T5ForConditionalGeneration

MODEL_NAME = "Salesforce/codet5-base-multi-sum"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Use the Tester class for evaluation
from scripts.training.tester import Tester

# You may need to adjust the device argument if not using GPU
tester = Tester("Salesforce/codet5-base-multi-sum", model, tokenizer, task="text2text-generation", device=0)

# Specify your test file path (should be a JSON with 'code' and 'doc' fields)
# test_file = "data/test_cleaned_declarative-7.json"
# test_file = "data/test_cleaned_declarative-10.json"
# test_file = "data/test_cleaned_declarative-100.json"
test_file = "data/test_cleaned_declarative.json"
tester.run(test_file, 10)
