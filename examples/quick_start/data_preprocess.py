# pip install datasets

from transformers import AutoTokenizer

model_name_or_path = r"E:\chz1\llm\LLaMA-Factory\saves\Qwen2-1.5B\freeze\stf\allclass_minus-atten"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

encoded_input = tokenizer("你是谁？")
print(encoded_input)