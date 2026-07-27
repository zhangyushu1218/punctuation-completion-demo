import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import f1_score, classification_report
import os

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 110M < 150M
RESUME_CHECKPOINT = "./punctuation_best"  # 纯模型权重，用于加载模型初始化，当设为空则从零初始化
TRAIN_CHECKPOINT = "./punctuation_model/checkpoint-xxx"  # 训练断点，仅续训用
DATASET_NAME = "SirlyDreamer/THUCNews"
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# 标点标签映射
LABELS = ["O", "COMMA", "PERIOD", "DUN"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# 目标标点符号
PUNCT_MAP = {
    "COMMA": "，",
    "PERIOD": "。",
    "DUN": "、"
}
TARGET_PUNCTS = set(["，", "。", "、"])


def clean_raw_text(text: str) -> str:
    """清洗原始新闻文本，保留汉字+目标标点，剔除其他符号"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。、]", "", text)
    text = re.sub(r"([，。、])+", r"\1", text)
    return text.strip()


def build_punct_label_sequence(text: str):
    """
    输入带标点文本，输出：无标点文本 + 对应标签序列
    例："今天，天气好。" -> src="今天天气好", label=["O","O","COMMA","O","O","O","PERIOD"]
    """
    src_chars = []
    label_seq = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in TARGET_PUNCTS:
            if len(src_chars) > 0:
                last_idx = len(src_chars) - 1
                punct_label = [k for k, v in PUNCT_MAP.items() if v == char][0]
                label_seq[last_idx] = punct_label
            i += 1
        else:
            src_chars.append(char)
            label_seq.append("O")
            i += 1
    src_text = "".join(src_chars) if src_chars else "无"
    if not label_seq:
        label_seq = ["O"]
    return src_text, label_seq


def tokenize_and_align_labels(examples, tokenizer):
    """将文本转换为token并对齐标签"""
    batch_src = []
    batch_char_tags = []

    for raw_text in examples["text"]:
        clean_t = clean_raw_text(raw_text)
        src, tag_list = build_punct_label_sequence(clean_t)
        batch_src.append(src)
        batch_char_tags.append(tag_list)

    token_out = tokenizer(
        batch_src,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_list = token_out.pop("offset_mapping")
    all_label_ids = []

    for idx in range(len(batch_src)):
        char_tags = batch_char_tags[idx]
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
        print("未检测到训练权重，从零初始化预训练BERT微调")
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


def post_process_punctuation(text: str) -> str:
    """
    后处理标点符号，修复常见模式

    规则：
    1. 修正数字后的顿号（12174列、货物 -> 12174列，货物）- 优先级最高
    2. 修复铁路干线名称（京、广、京、沪 -> 京广、京沪）
    3. 移除单字后的顿号（如：中、西部 -> 中西部）
    """
    import re

    # 规则1: 先修正"数量词 + 顿号 + 名词"的错误（应该是逗号）- 优先级最高
    # 例如："12174列、货物" -> "12174列，货物"
    # 必须在其他规则之前执行，避免顿号被提前移除
    number_dunhao = r'(\d+[\u4e00-\u9fa5]*)、([\u4e00-\u9fa5])'
    text = re.sub(number_dunhao, r'\1，\2', text)

    # 规则2: 处理铁路干线名称模式
    # 模式1: 带顿号的省份简称序列 "京、广、京、沪" -> "京广、京沪"
    railway_with_dunhao = r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])'
    
    def replace_railway_with_dunhao(match):
        """将'京、广、京、沪'转换为'京广、京沪'"""
        return f"{match.group(1)}{match.group(2)}、{match.group(3)}{match.group(4)}"
    
    text = re.sub(railway_with_dunhao, replace_railway_with_dunhao, text)
    
    # 模式2: 已经合并的铁路干线名称 "京广京沪" -> "京广、京沪"
    # 使用边界确保只匹配4个连续的省份简称字符
    railway_pattern = r'(?<![京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{2})([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{2})(?![京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])'
    
    def replace_railway(match):
        """将'京广京沪'转换为'京广、京沪'"""
        return f"{match.group(1)}、{match.group(2)}"
    
    text = re.sub(railway_pattern, replace_railway, text)

    # 规则3: 移除单字之间的顿号（过度预测的顿号）
    # 匹配模式：单字 + 顿号 + 单字，如"中、西"、"东、西"
    single_char_dunhao = r'([\u4e00-\u9fa5])、([\u4e00-\u9fa5])(?=[\u4e00-\u9fa5]|$)'

    def fix_single_char_dunhao(match):
        """移除单字之间的顿号"""
        char1 = match.group(1)
        char2 = match.group(2)

        # 检查是否是常见的单字省份简称组合（应该合并）
        province_chars = set('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
        if char1 in province_chars and char2 in province_chars:
            # 这是省份简称，应该合并（如"京广"、"京沪"）
            return f"{char1}{char2}"

        # 其他单字组合也默认合并（避免"中、西部"这样的错误）
        return f"{char1}{char2}"

    text = re.sub(single_char_dunhao, fix_single_char_dunhao, text)

    return text
