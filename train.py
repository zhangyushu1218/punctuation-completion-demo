import torch
import json
import os
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from utils import (
    clean_raw_text,
    tokenize_and_align_labels,
    compute_metrics,
    load_model_and_tokenizer,
    DATASET_NAME,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TRAIN_CHECKPOINT,
    LABEL2ID,
    NUM_LABELS,
    DEVICE
)


def train_model():
    """执行训练流程"""
    print("=" * 50)
    print("开始训练流程")
    print("=" * 50)

    # 加载数据集
    print("加载THUCNews数据集...")
    dataset = load_dataset(DATASET_NAME, split="train[:50000]")

    def filter_func(sample):
        t = clean_raw_text(sample["text"])
        return len(t) >= 8

    dataset = dataset.filter(filter_func)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    print(f"训练集:{len(train_ds)} 验证集:{len(val_ds)}")

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()

    # 数据预处理
    tokenize_fn = lambda x: tokenize_and_align_labels(x, tokenizer)
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_tokenized.set_format("python", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format("python", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 计算类别权重，解决类别不平衡问题（使用更温和的策略）
    from collections import Counter
    import numpy as np
    
    all_labels = []
    for example in train_tokenized:
        for label in example["labels"]:
            if label != -100:  # 排除padding
                all_labels.append(label)
    
    label_counts = Counter(all_labels)
    total_samples = sum(label_counts.values())
    num_classes = len(LABEL2ID)
    
    # 使用平方根缩放而非直接倒数，避免权重过大
    # 公式：weight = sqrt(total / (num_classes * count))
    class_weights = np.ones(num_classes)
    for label_idx, count in label_counts.items():
        if count > 0:
            # 使用平方根缩放，限制最大权重为3.0
            raw_weight = np.sqrt(total_samples / (num_classes * count))
            class_weights[label_idx] = min(raw_weight, 5.0)  # 设置上限
    
    print(f"\n类别分布: {dict(label_counts)}")
    print(f"原始权重: {[np.sqrt(total_samples / (num_classes * label_counts.get(i, 1))) for i in range(num_classes)]}")
    print(f"应用上限后的权重: {class_weights.tolist()}")
    print(f"DUN标签权重: {class_weights[LABEL2ID['DUN']]:.2f}x (上限3.0x)\n")
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir="./punctuation_model",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # 启动训练（使用类别权重）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 为损失函数设置类别权重
    if hasattr(trainer, 'compute_loss'):
        original_compute_loss = trainer.compute_loss
        
        def weighted_compute_loss(model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, NUM_LABELS)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            loss = loss_fct(active_logits, active_labels)
            
            if return_outputs:
                return loss, outputs
            return loss
        
        trainer.compute_loss = weighted_compute_loss

    print("开始训练...")
    has_train_ckpt = os.path.exists(os.path.join(TRAIN_CHECKPOINT, "trainer_state.json"))
    if has_train_ckpt:
        print(f"检测到训练断点，从 checkpoint 续训: {TRAIN_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=TRAIN_CHECKPOINT)
    else:
        trainer.train()

    # 验证集评测
    print("\n===== 验证集完整评测结果 =====")
    eval_result = trainer.evaluate()
    for k, v in eval_result.items():
        print(f"{k}: {v:.4f}")

    # 保存评估结果到文件
    eval_result_file = "./eval_results.json"
    eval_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "hfl/chinese-bert-wwm-ext",
        "dataset": DATASET_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in eval_result.items()}
    }
    
    # 如果文件已存在，读取历史结果并追加
    if os.path.exists(eval_result_file):
        with open(eval_result_file, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    
    history.append(eval_record)
    
    with open(eval_result_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存至: {eval_result_file}")
    print(f"历史记录数: {len(history)}")

    # 保存模型
    model.save_pretrained("./punctuation_best")
    tokenizer.save_pretrained("./punctuation_best")
    print("模型已保存至 ./punctuation_best")


if __name__ == "__main__":
    train_model()
