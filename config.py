"""
全局配置模块
包含模型、训练、数据处理等所有配置项
"""
import torch


# ===================== 设备配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 模型配置 =====================
# MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 110M < 150M
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 110M
RESUME_CHECKPOINT = "./punctuation_best"  # 纯模型权重，用于加载模型初始化，当设为空则从零初始化
TRAIN_CHECKPOINT = "./punctuation_model/checkpoint-xxx"  # 训练断点，仅续训用


# ===================== 数据集配置 =====================
DATASET_NAME = "SirlyDreamer/THUCNews"


# ===================== 训练超参数 =====================
MAX_SEQ_LEN = 512  # 最大序列长度（包含特殊token）
BATCH_SIZE = 32    # 批次大小
EPOCHS = 3         # 训练轮数
LEARNING_RATE = 2e-5  # 学习率


# ===================== 滑动窗口配置 =====================
USE_SLIDING_WINDOW = False   # 是否启用滑动窗口
SLIDING_WINDOW_STRIDE = 256  # 滑动步长


# ===================== 标签配置 =====================
# 标点标签映射（优化后：移除偏僻符号，合并相似符号）
LABELS = [
    "O",
    "COMMA",        # ，
    "PERIOD",       # 。 （包含：！、？、；）
    "DUN",          # 、
    "COLON",        # ：
    "QUOTE_LEFT",   # "
    "QUOTE_RIGHT",  # "
    "BRACKET_LEFT", # 【 （包含：（、《）
    "BRACKET_RIGHT" # 】 （包含：）、》）
]

# 标签与ID的映射
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
NUM_LABELS = len(LABELS)
