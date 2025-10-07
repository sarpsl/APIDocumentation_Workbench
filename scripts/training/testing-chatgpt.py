from peft import PeftModel

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./codet5-lora-adapter")

input_code = "def add(a, b): return a+b"
inputs = tokenizer(input_code, return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=64, num_beams=5)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

