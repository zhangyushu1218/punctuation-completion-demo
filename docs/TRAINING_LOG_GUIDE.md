# 训练日志系统使用说明

## 📋 概述

本次优化为ASR标点预测项目添加了完整的训练日志保存功能，所有训练参数、配置和结果都会自动保存到两个位置：

1. **日志文件** (`./logs/training_YYYYMMDD_HHMMSS.log`) - 详细的训练过程日志
2. **JSON记录** (`./eval_results.json`) - 结构化的训练结果汇总

---

## 🎯 保存的参数详情

### 1. 数据集信息 (dataset)
- 数据集名称
- 训练集样本数
- 验证集样本数
- 数据划分比例
- 随机种子（保证可复现性）

### 2. 超参数配置 (hyperparameters)
- Epochs（训练轮数）
- Batch size（批次大小）
- Learning rate（学习率）
- Max sequence length（最大序列长度）
- FP16混合精度开关
- Weight decay（权重衰减）
- Warmup ratio（学习率预热比例）

### 3. 模型架构信息 (model_architecture)
- 总参数量
- 可训练参数量
- 标签数量
- 标签映射关系 (label2id)

### 4. 类别权重策略 (class_weights)
- 权重计算策略描述
- 每个标签的具体权重值
- 类别分布统计

### 5. 评估指标 (metrics)
- Macro F1分数
- 字符准确率
- 各标签单独的F1分数
- 验证集损失

### 6. 训练环境信息 (training_info)
- 最佳epoch编号
- 训练总耗时（秒/分钟）
- 设备信息（GPU/CPU）
- CUDA可用性
- 日志文件路径

---

## 📁 文件结构

```
asr_demo/
├── logs/                          # 训练日志目录
│   ├── training_20260727_164500.log  # 按时间戳命名的日志文件
│   ├── training_20260728_093000.log
│   └── ...
├── eval_results.json              # 历史训练结果汇总（JSON数组）
├── train.py                       # 训练脚本（已优化）
└── utils.py                       # 工具函数
```

---

## 🔍 日志文件示例

每次训练会生成一个独立的日志文件，包含：

```
2026-07-27 16:45:00 - INFO - ================================================================================
2026-07-27 16:45:00 - INFO - 开始训练流程
2026-07-27 16:45:00 - INFO - ================================================================================
2026-07-27 16:45:05 - INFO - 训练集样本数: 45000
2026-07-27 16:45:05 - INFO - 验证集样本数: 5000
2026-07-27 16:45:08 - INFO - 模型总参数量: 102.27M (102,267,652)
2026-07-27 16:45:12 - INFO - 类别分布统计:
2026-07-27 16:45:12 - INFO -   O (ID=0): 5,234,567 samples (95.23%)
2026-07-27 16:45:12 - INFO -   COMMA (ID=1): 187,234 samples (3.41%)
2026-07-27 16:45:12 - INFO -   PERIOD (ID=2): 62,345 samples (1.13%)
2026-07-27 16:45:12 - INFO -   DUN (ID=3): 12,456 samples (0.23%)
2026-07-27 16:45:12 - INFO - 应用后的权重:
2026-07-27 16:45:12 - INFO -   O: 1.0000x
2026-07-27 16:45:12 - INFO -   COMMA: 2.3456x
2026-07-27 16:45:12 - INFO -   PERIOD: 3.4567x
2026-07-27 16:45:12 - INFO -   DUN: 5.0000x
2026-07-27 17:15:42 - INFO - 训练完成！总耗时: 1830.25 秒 (30.50 分钟)
2026-07-27 17:15:45 - INFO - macro_f1: 0.8756
2026-07-27 17:15:45 - INFO - 最佳Epoch: 2
```

---

## 📊 JSON记录示例

`eval_results.json` 文件包含所有历史训练的汇总：

```json
[
  {
    "timestamp": "2026-07-27 17:15:45",
    "model_name": "hfl/chinese-bert-wwm-ext",
    "dataset": {
      "name": "SirlyDreamer/THUCNews",
      "train_samples": 45000,
      "val_samples": 5000,
      "split_ratio": 0.9,
      "seed": 42
    },
    "hyperparameters": {
      "epochs": 3,
      "batch_size": 16,
      "learning_rate": 2e-05,
      "max_seq_length": 128,
      "fp16": true,
      "weight_decay": 0.0,
      "warmup_ratio": 0.0
    },
    "model_architecture": {
      "total_parameters": 102267652,
      "trainable_parameters": 102267652,
      "num_labels": 4,
      "label2id": {
        "O": 0,
        "COMMA": 1,
        "PERIOD": 2,
        "DUN": 3
      }
    },
    "class_weights": {
      "strategy": "sqrt(total / (num_classes * count)) with cap at 5.0",
      "weights": {
        "O": 1.0,
        "COMMA": 2.3456,
        "PERIOD": 3.4567,
        "DUN": 5.0
      },
      "distribution": {
        "O": 5234567,
        "COMMA": 187234,
        "PERIOD": 62345,
        "DUN": 12456
      }
    },
    "metrics": {
      "eval_loss": 0.0234,
      "macro_f1": 0.8756,
      "char_accuracy": 0.9823,
      "O_f1": 0.9912,
      "COMMA_f1": 0.8234,
      "PERIOD_f1": 0.8567,
      "DUN_f1": 0.7345
    },
    "training_info": {
      "best_epoch": 2,
      "duration_seconds": 1830.25,
      "duration_minutes": 30.5,
      "device": "cuda",
      "cuda_available": true,
      "log_file": "./logs/training_20260727_164500.log"
    }
  }
]
```

---

## 💡 使用优势

### 1. **实验追溯**
- 每次训练都有完整的时间戳和配置记录
- 可以快速定位某次实验的详细参数

### 2. **对比分析**
- 通过 `eval_results.json` 对比不同超参数的效果
- 分析类别权重对性能的影响

### 3. **性能调优**
- 追踪训练时长变化，识别瓶颈
- 分析类别不平衡问题的改善情况

### 4. **结果复现**
- 通过保存的随机种子和完整配置重现结果
- 日志文件提供详细的训练过程信息

### 5. **问题排查**
- 日志文件记录训练过程中的关键节点
- 便于定位训练异常或性能下降的原因

---

## 🚀 使用方法

直接运行训练脚本即可自动生成日志：

```bash
python train.py
```

训练完成后：
1. 查看 `./logs/` 目录下的最新日志文件
2. 查看 `./eval_results.json` 了解历史训练记录

---

## 📝 注意事项

1. **日志文件命名**：使用时间戳格式 `training_YYYYMMDD_HHMMSS.log`，确保唯一性
2. **JSON追加模式**：`eval_results.json` 采用数组格式，每次训练自动追加新记录
3. **日志级别**：当前设置为 `INFO`，可根据需要调整为 `DEBUG` 或 `WARNING`
4. **磁盘空间**：定期清理旧日志文件，避免占用过多存储空间

---

## 🔧 技术实现

### 核心组件

1. **setup_training_logger()** 函数
   - 创建 `./logs/` 目录
   - 生成带时间戳的日志文件名
   - 配置双输出（文件 + 控制台）
   - 设置UTF-8编码支持中文

2. **结构化日志记录**
   - 使用 `logger.info()` 替代 `print()`
   - 添加阶段标识（如【数据准备】、【模型加载】等）
   - 记录关键参数和统计信息

3. **JSON记录增强**
   - 分层组织参数（dataset, hyperparameters, metrics等）
   - 保存类别权重和分布详情
   - 记录训练时长和设备信息
   - 关联日志文件路径

---

## ✨ 总结

通过本次优化，训练流程现在能够：
- ✅ 自动保存完整的训练参数到日志文件
- ✅ 结构化存储历史训练记录到JSON
- ✅ 提供详细的训练过程追踪
- ✅ 支持实验对比和结果复现
- ✅ 便于性能分析和问题排查

这为后续的模型迭代和实验管理提供了坚实的基础！
