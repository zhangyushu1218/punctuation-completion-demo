import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import f1_score, classification_report
import os
from config import (
    DEVICE,
    MODEL_NAME,
    RESUME_CHECKPOINT,
    MAX_SEQ_LEN,
    LABELS,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    USE_SLIDING_WINDOW,
    SLIDING_WINDOW_STRIDE
)
from data_processor import (
    batch_process_samples,
    PUNCT_MAP, clean_raw_text, build_punct_label_sequence, PunctuationDataset,
)


def tokenize_and_align_labels(examples, tokenizer):
    """将文本转换为token并对齐标签（支持滑动窗口）"""
    # 使用数据处理器批量处理样本（包含滑动窗口）
    all_texts, all_labels = batch_process_samples(
        examples, 
        max_seq_len=MAX_SEQ_LEN,
        use_sliding_window=USE_SLIDING_WINDOW,
        stride=SLIDING_WINDOW_STRIDE
    )
    
    # Tokenizer 处理
    token_out = tokenizer(
        all_texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_list = token_out.pop("offset_mapping")
    all_label_ids = []

    for idx in range(len(all_texts)):
        char_tags = all_labels[idx]
        offsets = offset_list[idx]
        label_ids = []
        for (start, end) in offsets:
            if start == 0 and end == 0:
                label_ids.append(-100)
            elif start >= len(char_tags):
                label_ids.append(-100)
            else:
                tag = char_tags[start]
                label_ids.append(LABEL2ID[tag])
        if len(label_ids) < MAX_SEQ_LEN:
            label_ids += [-100] * (MAX_SEQ_LEN - len(label_ids))
        all_label_ids.append(label_ids)

    for arr in all_label_ids:
        for val in arr:
            assert isinstance(val, int), f"发现非数字标签: {val}"
    token_out["labels"] = all_label_ids
    return token_out


def compute_metrics(eval_pred):
    """
    评测指标：
    1. 整体标点F1（位置+类型）
    2. 各标点单独F1
    3. 字符级匹配准确率
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_all = []
    pred_all = []
    char_correct = 0
    char_total = 0

    for pred_seq, true_seq in zip(predictions, labels):
        for p, t in zip(pred_seq, true_seq):
            if t == -100:
                continue
            true_all.append(t)
            pred_all.append(p)
            char_total += 1
            if p == t:
                char_correct += 1

    char_acc = char_correct / char_total if char_total > 0 else 0.0
    total_f1 = f1_score(true_all, pred_all, average="macro")
    report = classification_report(
        true_all, pred_all, target_names=LABELS, output_dict=True, zero_division=0
    )
    res = {
        "macro_f1": total_f1,
        "char_accuracy": char_acc,
    }
    for lab in LABELS:
        res[f"{lab}_f1"] = report[lab]["f1-score"]
    return res


def load_model_and_tokenizer():
    """加载模型和分词器"""
    print("加载分词器和模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"检测到已训练权重，加载 checkpoint: {RESUME_CHECKPOINT}")
        model = AutoModelForTokenClassification.from_pretrained(
            RESUME_CHECKPOINT,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        ).to(DEVICE)
    else:
        print("未检测到训练权重，从零初始化预训练模型微调")
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    assert total_params < 150 * 1e6, "模型参数量超过150M限制！"

    return model, tokenizer


def infer_punctuation(raw_no_punc: str, model, tokenizer):
    """输入无标点ASR文本，输出带标点完整句子"""
    model.eval()
    token_res = tokenizer(
        raw_no_punc,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_offsets_mapping=True
    )
    offset_mapping = token_res.pop("offset_mapping")[0]
    inputs = {k: v.to(DEVICE) for k, v in token_res.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    char_with_punc = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue
        char = raw_no_punc[start:end]
        char_with_punc.append(char)
        pred_label = ID2LABEL[pred_ids[i]]
        if pred_label != "O":
            char_with_punc.append(PUNCT_MAP[pred_label])
    
    result = "".join(char_with_punc)
    
    return result

def prepare_dataset(dataset, tokenizer, use_sliding_window=True, stride=256):
    """
    准备数据集（支持滑动窗口）

    Args:
        dataset: HuggingFace Dataset对象
        tokenizer: HuggingFace tokenizer
        use_sliding_window: 是否使用滑动窗口
        stride: 滑动步长

    Returns:
        处理后的Dataset对象或PunctuationDataset对象
    """
    if use_sliding_window:
        # 使用滑动窗口处理
        texts = []
        labels = []
        for sample in dataset:
            clean_t = clean_raw_text(sample["text"])
            src_text, label_seq = build_punct_label_sequence(clean_t)
            texts.append(src_text)
            labels.append(label_seq)

        processed_dataset = PunctuationDataset(
            texts, labels, tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            stride=stride
        )
        return processed_dataset
    else:
        # 不使用滑动窗口，直接截断
        tokenize_fn = lambda x: tokenize_and_align_labels(x, tokenizer)
        processed_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        processed_dataset.set_format("python", columns=["input_ids", "attention_mask", "labels"])
        return processed_dataset