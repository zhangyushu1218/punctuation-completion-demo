"""
标点符号预测 - 主程序入口

通过 MODE 参数控制执行训练或推理流程：
- MODE = 'train': 执行训练流程
- MODE = 'inference': 执行推理流程
"""

from train import train_model
from inference import inference_model

# ===================== 模式控制 =====================
# 固定参数：'train' 或 'inference'
# MODE = 'train'
MODE = 'inference'


if __name__ == "__main__":
    print(f"当前运行模式: {MODE}\n")

    if MODE == 'train':
        train_model()
    elif MODE == 'inference':
        inference_model()
    else:
        raise ValueError(f"不支持的模式: {MODE}，请使用 'train' 或 'inference'")
