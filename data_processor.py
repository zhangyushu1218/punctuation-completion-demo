"""
数据预处理模块
负责文本清洗、标签序列构建、滑动窗口分片等功能
"""
import re
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import torch
from config import LABEL2ID

# ===================== 标点符号配置 =====================
# 偏僻符号到常见符号的映射（训练时使用）
PUNCT_NORMALIZATION = {
    "！": "。",  # EXCLAMATION -> PERIOD
    "？": "。",  # QUESTION -> PERIOD
    "；": "。",  # SEMICOLON -> PERIOD
    "（": "【",  # PAREN_LEFT -> BRACKET_LEFT
    "）": "】",  # PAREN_RIGHT -> BRACKET_RIGHT
    "《": "【",  # BOOK_LEFT -> BRACKET_LEFT
    "》": "】",  # BOOK_RIGHT -> BRACKET_RIGHT
}

# 目标标点符号（标准化后的）
PUNCT_MAP = {
    "COMMA": "，",
    "PERIOD": "。",
    "DUN": "、",
    "COLON": "：",
    "QUOTE_LEFT": "\u201c",
    "QUOTE_RIGHT": "\u201d",
    "BRACKET_LEFT": "\u3010",
    "BRACKET_RIGHT": "\u3011"
}

TARGET_PUNCTS = {
    "，", "。", "、", "：",
    "\u201c", "\u201d",
    "\u3010", "\u3011"
}


def clean_raw_text(text: str) -> str:
    """清洗原始新闻文本，保留汉字+目标标点（含归一化标点），剔除其他符号
    
    支持的标点：
    - 基础标点：逗号(，)、句号(。)、顿号(、)、冒号(：)
    - 引号：左双引号(")、右双引号(")
    - 括号：左中括号(【)、右中括号(】)
    - 归一化标点（训练时会被转换）：感叹号(！)、问号(？)、分号(；)、左括号(（)、右括号(）)、左书名号(《)、右书名号(》)
    """
    # 保留汉字、字母、数字以及所有可能的标点符号（包括需要归一化的）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\uFF0C\u3002\u3001\uFF1A\u201C\u201D\u3010\u3011\uFF01\uFF1F\uFF1B\uFF08\uFF09\u300A\u300B]", "", text)
    # 去重连续相同标点
    text = re.sub(r"([\uFF0C\u3002\u3001\uFF1A\u201C\u201D\u3010\u3011\uFF01\uFF1F\uFF1B\uFF08\uFF09\u300A\u300B])\1+", r"\1", text)
    return text.strip()


def build_punct_label_sequence(text: str):
    """
    输入带标点文本，输出：无标点文本 + 对应标签序列
    例："今天，天气好。" -> src="今天天气好", label=["O","O","COMMA","O","O","O","PERIOD"]
    支持的标点：逗号(，)、句号(。)、顿号(、)、冒号(：)、双引号("/")、中括号(【】)
    
    符号归一化策略（训练时）：
    - 感叹号(！)、问号(？)、分号(；) -> 句号(。)
    - 左括号(（)、左书名号(《) -> 左中括号(【)
    - 右括号(）)、右书名号(》) -> 右中括号(】)
    - 波浪号(~) -> 移除
    
    关键设计：
    - 标点符号从 src 中移除，不占用独立位置
    - 标点标签标记在前一个字符上（会覆盖该字符原有的"O"标签）
    - 连续标点时，后一个标点的标签会覆盖前一个
    - 最终：len(src) == len(label_seq)
    """
    src_chars = []
    label_seq = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in TARGET_PUNCTS or char in PUNCT_NORMALIZATION:
            # 标点符号：标记在前一个字符上
            if len(src_chars) > 0:
                last_idx = len(src_chars) - 1
                # 先进行符号归一化
                normalized_char = PUNCT_NORMALIZATION.get(char, char)
                # 查找对应的标签
                punct_label = [k for k, v in PUNCT_MAP.items() if v == normalized_char][0]
                label_seq[last_idx] = punct_label  # 覆盖前一个字符的标签
            i += 1
        else:
            # 普通字符：添加到 src 并标记为"O"
            src_chars.append(char)
            label_seq.append("O")
            i += 1
    
    src_text = "".join(src_chars) if src_chars else "无"
    if not label_seq:
        label_seq = ["O"]
    
    # 确保长度一致
    assert len(src_text) == len(label_seq), f"标签序列长度({len(label_seq)})与文本长度({len(src_text)})不匹配！"
    
    return src_text, label_seq


def create_sliding_window_segments(
    text: str, 
    labels: List[str], 
    max_length: int = 512, 
    stride: int = 256
) -> List[Tuple[str, List[str]]]:
    """
    使用滑动窗口将长文本切分为多个片段
    
    Args:
        text: 无标点文本
        labels: 对应的标签序列
        max_length: 最大序列长度（包含特殊token [CLS] 和 [SEP]）
        stride: 滑动步长
    
    Returns:
        列表，每个元素为 (text_segment, label_segment) 元组
    
    示例：
        text = "abcdefghij"
        labels = ["O", "O", "COMMA", "O", "O", "O", "PERIOD", "O", "O", "O"]
        max_length = 5, stride = 3
        
        返回：
        [
            ("abcde", ["O", "O", "COMMA", "O", "O"]),
            ("cdefg", ["COMMA", "O", "O", "O", "PERIOD"]),
            ("efghi", ["O", "O", "PERIOD", "O", "O"]),
            ("ghij", ["PERIOD", "O", "O", "O"])
        ]
    """
    segments = []
    
    # 如果文本长度不超过限制，直接返回
    if len(text) <= max_length - 2:  # 减去 [CLS] 和 [SEP]
        return [(text, labels)]
    
    # 计算实际可用的内容长度（扣除特殊token）
    content_max_len = max_length - 2
    
    # 滑动窗口切分
    start = 0
    while start < len(text):
        end = min(start + content_max_len, len(text))
        
        # 提取当前窗口的文本和标签
        segment_text = text[start:end]
        segment_labels = labels[start:end]
        
        segments.append((segment_text, segment_labels))
        
        # 如果已经到达末尾，退出
        if end >= len(text):
            break
        
        # 移动窗口
        start += stride
    
    return segments


def process_single_sample(
    raw_text: str, 
    max_seq_len: int = 512, 
    use_sliding_window: bool = True,
    stride: int = 256
) -> List[Tuple[str, List[str]]]:
    """
    处理单个样本：清洗文本 -> 构建标签序列 -> 滑动窗口分片
    
    Args:
        raw_text: 原始文本
        max_seq_len: 最大序列长度
        use_sliding_window: 是否使用滑动窗口
        stride: 滑动步长
    
    Returns:
        列表，每个元素为 (text_segment, label_segment) 元组
    """
    # 1. 清洗文本
    cleaned_text = clean_raw_text(raw_text)
    
    # 2. 构建标签序列
    src_text, label_seq = build_punct_label_sequence(cleaned_text)
    
    # 3. 滑动窗口分片（如果需要）
    if use_sliding_window:
        segments = create_sliding_window_segments(
            src_text, label_seq, max_seq_len, stride
        )
    else:
        # 不使用滑动窗口，直接截断
        if len(src_text) > max_seq_len - 2:
            src_text = src_text[:max_seq_len - 2]
            label_seq = label_seq[:max_seq_len - 2]
        segments = [(src_text, label_seq)]
    
    return segments


def batch_process_samples(
    examples: Dict[str, List[str]], 
    max_seq_len: int = 512,
    use_sliding_window: bool = True,
    stride: int = 256
) -> Tuple[List[str], List[List[str]]]:
    """
    批量处理样本
    
    Args:
        examples: 包含 "text" 字段的字典，值为文本列表
        max_seq_len: 最大序列长度
        use_sliding_window: 是否使用滑动窗口
        stride: 滑动步长
    
    Returns:
        (所有文本片段列表, 所有标签序列列表)
    """
    all_texts = []
    all_labels = []
    
    for raw_text in examples["text"]:
        segments = process_single_sample(
            raw_text, max_seq_len, use_sliding_window, stride
        )
        for text_seg, label_seg in segments:
            all_texts.append(text_seg)
            all_labels.append(label_seg)
    
    return all_texts, all_labels


class PunctuationDataset(Dataset):
    """
    支持滑动窗口的标点预测数据集
    
    当文本超过max_seq_len时，使用滑动窗口将其切分为多个片段，
    每个片段保持max_seq_len长度，相邻片段之间有stride的重叠。
    """
    
    def __init__(self, texts: List[str], labels: List[List[str]], 
                 tokenizer, max_seq_len: int = 512, stride: int = 256):
        """
        Args:
            texts: 无标点文本列表
            labels: 对应的标签序列列表
            tokenizer: HuggingFace tokenizer
            max_seq_len: 最大序列长度（包含特殊token）
            stride: 滑动步长
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        
        # 存储所有样本（经过滑动窗口切分后）
        self.samples = []
        
        # 处理每个原始样本
        for text, label_seq in zip(texts, labels):
            self._process_sample(text, label_seq)
    
    def _process_sample(self, text: str, label_seq: List[str]):
        """处理单个样本，如果过长则使用滑动窗口切分"""
        # 先进行tokenization看看长度
        tokens = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_offsets_mapping=True
        )
        
        input_ids = tokens['input_ids']
        offset_mapping = tokens['offset_mapping']
        
        # 如果不超过max_seq_len，直接添加
        if len(input_ids) <= self.max_seq_len:
            self.samples.append({
                'text': text,
                'label_seq': label_seq,
                'offset_mapping': offset_mapping,
                'is_windowed': False
            })
        else:
            # 使用滑动窗口切分
            # 减去[CLS]和[SEP]的占用
            content_max_len = self.max_seq_len - 2
            
            start_pos = 0
            text_len = len(text)
            
            while start_pos < text_len:
                # 计算当前窗口的结束位置
                end_pos = min(start_pos + content_max_len, text_len)
                
                # 提取当前窗口的文本
                window_text = text[start_pos:end_pos]
                
                # 提取对应的标签序列
                # 需要找到start_pos和end_pos在原始label_seq中的位置
                # 这里简化处理：假设字符级别对齐
                window_labels = label_seq[start_pos:end_pos]
                
                # 对这个窗口进行tokenization
                window_tokens = self.tokenizer(
                    window_text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding='max_length',
                    return_offsets_mapping=True
                )
                
                self.samples.append({
                    'text': window_text,
                    'label_seq': window_labels,
                    'offset_mapping': window_tokens['offset_mapping'],
                    'is_windowed': True,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                })
                
                # 如果已经到达末尾，退出
                if end_pos >= text_len:
                    break
                
                # 移动窗口
                start_pos += self.stride
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 构建标签
        label_seq = sample['label_seq']
        offset_mapping = encoding['offset_mapping'][0]
        
        label_ids = []
        for (start, end) in offset_mapping:
            if start == 0 and end == 0:
                # Padding或特殊token
                label_ids.append(-100)
            elif start >= len(label_seq):
                label_ids.append(-100)
            else:
                tag = label_seq[start]
                label_ids.append(LABEL2ID[tag])
        
        # 确保长度一致
        if len(label_ids) < self.max_seq_len:
            label_ids += [-100] * (self.max_seq_len - len(label_ids))
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
