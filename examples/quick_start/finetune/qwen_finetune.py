import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab

from transformers.models.qwen2.modular_qwen2 import Qwen2ForCausalLM

prompt = """以下是一个中学网上微课堂的音频转录文本。请首先简述课堂内容并说明对应的学科知识体系，然后根据以下维度对该课堂进行批判性的分析评价：
1.教学内容是否有合理的引入、展开、总结。讲解是否有条理，层次分明，过渡自然。
2.内容的逻辑性和条理性，讲解是否循序渐进，是否存在内容跳跃或逻辑不清晰的情况。
3.关键概念和知识点是否被充分突出，特别是教学中的重点与难点。
4.语言是否简洁、明了、规范，避免过于复杂或冗长的句子。语言是否符合学生的认知水平，是否能准确表达所需的知识。
5.在转录文本中查找是否有引导学生思考的问题或提示，是否有针对性的互动设计。
6.从转录文本中评估教学方法是否有所创新，比如是否运用了新的教学策略或形式。
7.检查转录文本结尾是否有总结和回顾所学内容，是否提供了延伸阅读或思考的问题，鼓励学生对内容进行深入反思。
请注意：转录的文本本身就存在着一些错误，请忽略那些可能因为转录导致的错误。
"""
# def dataset_jsonl_transfer(origin_path, new_path):
#     """
#     将原始数据集转换为大模型微调所需数据格式的新数据集
#     """
#     messages = []
#     # 读取旧的JSONL文件
#     with open(origin_path, "r", encoding="utf-8") as file:
#         datas = json.load(file)
#         for data in datas:
#             context = data["text"]
#             label = data["eval"]
#             message = {
#                 "instruction": prompt,
#                 "input": context,
#                 "output": label,
#             }
#             messages.append(message)
#
#     # 保存重构后的JSONL文件
#     with open(new_path, "w", encoding="utf-8") as file:
#         for message in messages:
#             file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 4096
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


# 在modelscope上下载Qwen模型到本地目录下
# model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")
model_dir = r"D:\download\model\Qwen\Qwen2-1___5B-Instruct"
# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 冻结所有层的参数
for param in model.parameters():
    param.requires_grad = False

# 解冻模型中与注意力层相关的参数
for name, module in model.named_modules():
    # 根据层名称来解冻注意力层
    if 'self_attn' in name:  # 假设注意力层名称中包含 'attn'
        for param in module.parameters():
            param.requires_grad = True
# 验证冻结状态（可选）
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
print(f"Trainable parameters: {trainable_params}")
print(f"Frozen parameters: {frozen_params}")

# 加载、处理数据集和测试集
train_dataset_path = r"E:\chz1\llm\transformers\examples\quick_start\data\class_train.json"
test_dataset_path = r"E:\chz1\llm\transformers\examples\quick_start\data\class_val.json"

train_jsonl_new_path = r"E:\chz1\llm\transformers\examples\quick_start\data\new_train.json"
test_jsonl_new_path = r"E:\chz1\llm\transformers\examples\quick_start\data\new_val.json"

# if not os.path.exists(train_jsonl_new_path):
#     dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
# if not os.path.exists(test_jsonl_new_path):
#     dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,  # 训练模式
#     r=8,  # Lora 秩
#     lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.1,  # Dropout 比例
# )

# model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# swanlab_callback = SwanLabCallback(
#     project="Qwen2-fintune",
#     experiment_name="Qwen2-1.5B-Instruct",
#     description="使用通义千问Qwen2-1.5B-Instruct模型在class数据集上微调。",
#     config={
#         "model": "qwen/Qwen2-1.5B-Instruct",
#         "dataset": "class",
#     }
# )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    # callbacks=[swanlab_callback],
)

trainer.train()

# # 用测试集的前10条，测试模型
# test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]
#
# test_text_list = []
# for index, row in test_df.iterrows():
#     instruction = row['instruction']
#     input_value = row['input']
#
#     messages = [
#         {"role": "system", "content": f"{instruction}"},
#         {"role": "user", "content": f"{input_value}"}
#     ]
#
#     response = predict(messages, model, tokenizer)
#     messages.append({"role": "assistant", "content": f"{response}"})
#     result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
#     test_text_list.append(swanlab.Text(result_text, caption=response))
#
# swanlab.log({"Prediction": test_text_list})
# swanlab.finish()
