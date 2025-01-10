import os
import datasets
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

PaddingID = -100

def preprocess_inputs(examples, max_len=4096, overflow_strategy='truncate'):
    """
    @function:预处理输入数据
    examples：数据集
    max_len：Qwen2-7B-Instruct 支持 131072 tokens. max_len 的设置可以通过统计数据集的 token 长度得到。具体方法：将所有数据输入到 qwen2 模型的 tokenizer，统计 tokenizer 的输出长度（最大，最小，平均）
    overflow_strategy：'drop'表示丢弃，'truncate'表示截断
    """

    # prompt_template：可以在 qwen2 的 huggingface 官方库的 demo 中使用 tokenizer.apply_chat_template 函数打印模型的 prompt template
    # Qwen2-7B-Instruct huggingface 地址：https://huggingface.co/Qwen/Qwen2-7B-Instruct
    system_prompt = "你是一个知识渊博的人，请根据问题做出全面且正确的回答。"
    prompt_template = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'

    model_inputs = {'input_ids': [], 'labels': [], 'input_len': [], 'output_len': []}
    for i in range(len(examples['query'])):
        prompt = prompt_template.format(
                    system_prompt=system_prompt,
                    user_prompt=examples['query'][i]
                )
        a_ids = tokenizer.encode(prompt)
        b_ids = tokenizer.encode(f"{examples['answer'][i]}", add_special_tokens=False) + [tokenizer.eos_token_id]
        context_length = len(a_ids)
        input_ids = a_ids + b_ids

        if len(input_ids) > max_len and overflow_strategy == 'drop':
            # 丢弃样本
            input_ids = []
            labels = []
        else:
            if max_len > len(input_ids):
                """
                使用 -100 填充, 因为 torch.nn.CrossEntropyLoss 的 ignore_index=-100, 即 CrossEntropyLoss 会忽略标签为 -100 的值的 loss，只计算非填充部分的 loss
                torch.nn.CrossEntropyLoss 官方文档：https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                """
                pad_length = max_len - len(input_ids)
                labels = [PaddingID] * context_length + b_ids + [PaddingID] * pad_length
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            else:
                # 超过最大长度的数据被截断
                labels = [PaddingID] * context_length + b_ids
                labels = labels[:max_len]
                input_ids = input_ids[:max_len]
        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['input_len'].append(len(a_ids))
        model_inputs['output_len'].append(len(b_ids))
    return model_inputs

if __name__=="__main__":
    # load tokenizer
    model_path = 'models/qwen/Qwen2-7B-Instruct/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # load training dataset
    dataset_folder_path='sft_qwen2_7B/huggingface_data/'
    raw_datasets = load_dataset(dataset_folder_path)
    train_dataset = raw_datasets['train'].map(preprocess_inputs, batched=True, num_proc=1, load_from_cache_file=False)
    """
    加载 train.xxx 文件,如 train.txt  train.jsonl
    batched：分批加载数据，默认 batch=1000
    num_proc：配置多线程处理，一般不设置单线程的数据加载速度也很快
    load_from_cache_file：指定是否从缓存文件加载预处理后的数据。如果设置为 True，datasets 库会尝试从磁盘加载预先处理并缓存的数据集，而不是重新运行 map 函数。
                        设置 load_from_cache_file=False 意味着每次运行脚本时都会重新进行数据预处理，而不是从缓存中加载。
    """

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16,
                                                device_map='auto',
                                                trust_remote_code=True)
    """
    device_map 的可选参数有：auto、balanced、balanced_low_0、sequential
    'auto' 和 'balanced'是一样的，会自动切分模型，导致不同GPU之间的负载差异很大，可能某个卡的显存被占满了，其他卡只占了1/4，这会导致 batch_size 无法增大。
                        在训练qwen2-7B中，最后一个 GPU 占据资源很小（80G A100只占用 400M显存），其他GPU显存占用 20G～75G 不等。
    balanced_low_0：第一个 GPU 上占据较少资源（执行generate 函数，即迭代过程），其他 GPU 自动划分模型，也会负载不均。
    sequential：按照 GPU 的顺序分配模型分片，会导致 GPU 0 显存爆炸。
    综上来看，这四个参数都会使多卡 GPU 的负载不均，暂时没有发现如何能够平衡负载。
    """

    model.gradient_checkpointing_enable()
    """
    启用模型的梯度检查点, 梯度检查点是一种优化技术，可用于减少训练时的内存消耗。
    在反向传播期间，模型的中间激活值需要被保留以计算梯度。
    梯度检查点技术通过仅保存必要的一部分激活值，并在需要时重新计算丢弃的激活值，从而减少内存使用。
    """

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                        model=model,
                                        label_pad_token_id=PaddingID,
                                        pad_to_multiple_of=None,
                                        padding=False)
    """
    创建了一个 Seq2Seq任务 的数据整理器, 用于将多个样本组合成一个批次。
    label_pad_token_id：指定用于填充标签的 padding token 的 id, 默认为-100
    pad_to_multiple_of = None：指定padding后序列长度应该是多少的倍数。如果设置为None（默认值），则不进行这种类型的padding。
    padding = False：指定是否对数据进行padding。设置为False 通常意味着数据的 padding 将在模型内部或通过其他方式处理。
    """

    # 训练参数
    args = TrainingArguments(
        output_dir='./outputs',             # 模型保存路径
        per_device_train_batch_size=4,      # 全局 batch_size，注意不是单个 GPU 上的  batch_size
        logging_steps=1,
        gradient_accumulation_steps=32,     # 梯度累计，在显存较小的设备中，每隔多个 batch_size 更新一次梯度；
                                            # 真正更新梯度的 batch = per_device_train_batch_size * gradient_accumulation_steps
                                            # 即 4*32=128 个 batch 更新一次梯度
        num_train_epochs=1,                 # sft llm 的 epoch 一般不需要太大，1～3轮即可
        weight_decay=0.003,                 # 权重衰减正则化，将一个与权重向量的L2范数成比例的惩罚项加到总损失中
        warmup_ratio=0.03,                  # 预热，在训练初期逐渐增加学习率，而不是从一开始就使用预设的最大学习率，避免一开始就使用过高的学习率可能导致的训练不稳定。
                                            # 如果设置 warmup_ratio=0.1，共有100个epochs，那么在前10个epochs（即前10%的训练时间），学习率会从0逐渐增加到最大值。
        optim='adamw_hf',
        lr_scheduler_type="cosine",         # 根据余弦函数的形状来逐渐减小学习率，一般有 "linear" 和 "cosine" 两种方式
        learning_rate=5e-5,                 # 最大学习率
        save_strategy='steps',
        save_steps=5,                       # 保存模型的步骤，save_steps 是 per_device_train_batch_size * gradient_accumulation_steps，而不是 per_device_train_batch_size
        bf16=True,                       # 是否使用 bfloat16 数据格式
        run_name='qwen2-1.5B-sft',
        report_to='wandb',                  # 使用 wandb 打印日志
    )

    # train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
