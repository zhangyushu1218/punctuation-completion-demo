import torch
import json
import os
import time
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from training_logger import setup_training_logger, TrainingLogCallback
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
    DEVICE,
    MAX_SEQ_LEN,
    MODEL_NAME,
    LABELS
)


def train_model():
    """执行训练流程"""
    # 初始化日志系统
    logger, log_file = setup_training_logger()
    
    logger.info("=" * 80)
    logger.info("开始训练流程")
    logger.info("=" * 80)
    
    start_time = time.time()

    # 加载数据集
    logger.info(f"\n【数据准备】加载THUCNews数据集...")
    dataset = load_dataset(DATASET_NAME, split="train[:50000]")

    def filter_func(sample):
        t = clean_raw_text(sample["text"])
        return len(t) >= 8

    dataset = dataset.filter(filter_func)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    logger.info(f"训练集样本数: {len(train_ds)}")
    logger.info(f"验证集样本数: {len(val_ds)}")
    logger.info(f"数据划分比例: 90% 训练 / 10% 验证")
    logger.info(f"随机种子: 42")

    # 加载模型和分词器
    logger.info(f"\n【模型加载】加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params / 1e6:.2f}M ({total_params:,})")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M ({trainable_params:,})")

    logger.info(f"\n【数据处理】数据预处理与tokenization...")
    tokenize_fn = lambda x: tokenize_and_align_labels(x, tokenizer)
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_tokenized.set_format("python", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format("python", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    logger.info(f"最大序列长度: {MAX_SEQ_LEN}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    # 计算类别权重，解决类别不平衡问题（使用更温和的策略）
    from collections import Counter
    import numpy as np
    
    logger.info(f"\n【类别权重】分析类别分布并计算权重...")
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
            # 使用平方根缩放，限制最大权重为5.0
            raw_weight = np.sqrt(total_samples / (num_classes * count))
            class_weights[label_idx] = min(raw_weight, 5.0)  # 设置上限
    
    logger.info(f"类别分布统计:")
    for label, count in sorted(label_counts.items()):
        label_name = list(LABEL2ID.keys())[list(LABEL2ID.values()).index(label)]
        percentage = count / total_samples * 100
        logger.info(f"  {label_name} (ID={label}): {count:,} samples ({percentage:.2f}%)")
    
    logger.info(f"\n类别权重策略: sqrt(total / (num_classes * count)) with cap at 5.0")
    logger.info(f"应用后的权重:")
    for i, lab in enumerate(LABELS):
        logger.info(f"  {lab}: {class_weights[i]:.4f}x")
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    # 训练参数配置
    logger.info(f"\n【训练配置】设置训练参数...")
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
    
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Batch size (per device): {BATCH_SIZE}")
    logger.info(f"FP16混合精度: {torch.cuda.is_available()}")
    logger.info(f"Weight decay: {training_args.weight_decay}")
    logger.info(f"Warmup ratio: {training_args.warmup_ratio}")
    logger.info(f"设备: {DEVICE}")

    # 启动训练（使用类别权重）
    # 创建训练日志回调
    training_log_callback = TrainingLogCallback(logger)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[training_log_callback]
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
    logger.info(f"\n{'='*80}")
    logger.info("【训练阶段】开始模型训练...")
    logger.info(f"{'='*80}")
    has_train_ckpt = os.path.exists(os.path.join(TRAIN_CHECKPOINT, "trainer_state.json"))
    if has_train_ckpt:
        logger.info(f"检测到训练断点，从 checkpoint 续训: {TRAIN_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=TRAIN_CHECKPOINT)
    else:
        logger.info("从头开始训练")
        trainer.train()
    
    training_duration = time.time() - start_time
    logger.info(f"\n训练完成！总耗时: {training_duration:.2f} 秒 ({training_duration/60:.2f} 分钟)")

    # 验证集评测
    logger.info(f"\n{'='*80}")
    logger.info("【验证集评测】完整评测结果")
    logger.info(f"{'='*80}")
    eval_result = trainer.evaluate()
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")

    # 获取最佳epoch
    best_epoch = None
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_global_step'):
        best_step = trainer.state.best_global_step
        if best_step:
            steps_per_epoch = len(train_tokenized) // BATCH_SIZE
            best_epoch = best_step // steps_per_epoch + 1
            logger.info(f"\n最佳Epoch: {best_epoch}")

    # 保存评估结果到文件
    logger.info(f"\n{'='*80}")
    logger.info("【结果保存】保存训练记录和日志")
    logger.info(f"{'='*80}")
    
    eval_result_file = "./eval_results.json"
    
    # 构建完整的训练记录
    eval_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "dataset": {
            "name": DATASET_NAME,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "split_ratio": 0.9,
            "seed": 42
        },
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LEN,
            "fp16": torch.cuda.is_available(),
            "weight_decay": training_args.weight_decay,
            "warmup_ratio": training_args.warmup_ratio
        },
        "model_architecture": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_labels": NUM_LABELS,
            "label2id": LABEL2ID
        },
        "class_weights": {
            "strategy": "sqrt(total / (num_classes * count)) with cap at 5.0",
            "weights": {lab: round(float(class_weights[i]), 4) for i, lab in enumerate(LABELS)},
            "distribution": {lab: int(label_counts.get(i, 0)) for i, lab in enumerate(LABELS)}
        },
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in eval_result.items()},
        "training_process": training_log_callback.get_logs_summary(),
        "training_info": {
            "best_epoch": best_epoch,
            "duration_seconds": round(training_duration, 2),
            "duration_minutes": round(training_duration / 60, 2),
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "log_file": log_file
        }
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
    
    logger.info(f"\n评估结果已保存至: {eval_result_file}")
    logger.info(f"历史记录数: {len(history)}")
    logger.info(f"训练日志已保存至: {log_file}")

    # 保存模型
    logger.info(f"\n【模型保存】保存最佳模型...")
    model.save_pretrained("./punctuation_best")
    tokenizer.save_pretrained("./punctuation_best")
    logger.info(f"模型已保存至: ./punctuation_best")
    
    logger.info(f"\n{'='*80}")
    logger.info("训练流程全部完成！")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    train_model()
