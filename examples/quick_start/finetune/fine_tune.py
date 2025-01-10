import lawrouge
import numpy as np
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    set_seed,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, Qwen2ForCausalLM,
)

# Set seed before initializing model.
set_seed(42)

# 加载数据集
train_dataset = load_dataset('json', data_files='./data/class_val.json')

model_name = r"D:\download\model\Qwen\Qwen2-1___5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(model_name)

# Data collator
label_pad_token_id = tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)

max_input_length = 4096
max_target_length = 2048

batch_size = 1
args = Seq2SeqTrainingArguments(
    output_dir="../output/",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    eval_strategy="epoch",
    save_total_limit=3,
    # generation_max_length最大生成长度，系统默认20 generation_num_beams=1表示贪心解码，大于1为树搜索
    generation_max_length=2048,
    generation_num_beams=1,
)

def preprocess_function1(examples):
    examples["text"] = [prompt + "\n" + text for prompt, text in zip(examples["prompt"], examples["text"])]
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["eval"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

prefix = ("以下是一个中学网上微课堂的音频转录文本。请首先简述课堂内容并说明对应的学科知识体系，然后根据以下维度对该课堂进行批判性的分析评价："
          "1.教学内容是否有合理的引入、展开、总结。讲解是否有条理，层次分明，过渡自然。"
          "2.内容的逻辑性和条理性，讲解是否循序渐进，是否存在内容跳跃或逻辑不清晰的情况。"
          "3.关键概念和知识点是否被充分突出，特别是教学中的重点与难点。"
          "4.语言是否简洁、明了、规范，避免过于复杂或冗长的句子。语言是否符合学生的认知水平，是否能准确表达所需的知识。"
          "5.在转录文本中查找是否有引导学生思考的问题或提示，是否有针对性的互动设计。"
          "6.从转录文本中评估教学方法是否有所创新，比如是否运用了新的教学策略或形式。"
          "7.检查转录文本结尾是否有总结和回顾所学内容，是否提供了延伸阅读或思考的问题，鼓励学生对内容进行深入反思。"
        "请注意：转录的文本本身就存在着一些错误，请忽略那些可能因为转录导致的错误。")
def preprocess_function(examples):
    # remove pairs where at least one record is None
    inputs, targets = [], []
    for i in range(len(examples["text"])):
        if examples["text"][i] and examples["eval"][i]:
            inputs.append(examples["text"][i])
            targets.append(examples["eval"][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
def main():
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    # tokenized_val_datasets = valid_dataset.map(preprocess_function, batched=True)

    # trainer = SFTTrainer(
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_train_datasets["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # neftune_noise_alpha=5  ############SFTTrainer
    )

    # 评估时：
    eval_result = trainer.evaluate()
    print(eval_result)

    # 训练时：
    # train_result = trainer.train(resume_from_checkpoint=True)
    # train_result = trainer.train()
    # print(train_result)
    #
    # trainer.save_model()
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

# 这里用的是中文lawrouge 至于字符级还是词级计算看自己调整 这里是字符级
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 保存修改后的数据回JSON文件
    # import json
    # with open('./data/segment_model_output2.json', 'w', encoding='utf-8') as file:
    #     for i, entry in enumerate(decoded_preds):
    #         # 将JSON对象转化为字符串，并逐行写入文件
    #         result = {
    #             "summary": decoded_labels[i],
    #             "summary2": entry
    #         }
    #         json_str = json.dumps(result, ensure_ascii=False)
    #         file.write(json_str + '\n')

    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    # reserve_decoded_preds = [pred[::-1] for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]

    rouge = lawrouge.Rouge()

    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    print(result)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

    result = {key: value * 100 for key, value in result.items()}
    return result

if __name__ == "__main__":
    main()