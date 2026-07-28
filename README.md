# ASR标点预测项目

基于 BERT 的中文标点符号预测系统，支持训练、推理和完整的日志记录功能。

## 📁 项目结构

```
asr_demo/
├── train.py                    # 训练流程模块
├── inference.py                # 推理流程模块
├── utils.py                    # 通用工具函数和配置
├── training_logger.py          # 训练日志模块（独立）
├── eval_results.json           # 训练评估结果记录（自动生成）
├── requirements.txt            # 项目依赖
├── README.md                   # 项目说明文档
│
├── docs/                       # 📚 文档目录
│   ├── TRAINING_LOGGER_MODULE.md    # 日志模块使用手册
│   ├── TRAINING_LOG_GUIDE.md        # 训练日志系统指南
│   ├── TRAINING_PROGRESS_LOG.md     # 训练进度记录说明
│   ├── IMPLEMENTATION_SUMMARY.md    # 功能实现总结
│   └── REFACTORING_SUMMARY.md       # 代码重构总结
│
├── logs/                       # 📝 训练日志目录（自动生成）
│   └── training_YYYYMMDD_HHMMSS.log
│
├── punctuation_best/           # 🤖 最佳模型保存目录
└── punctuation_model/          # 🔄 训练checkpoint目录
```

> **注意**: `demo_punctuation.py` 已移除，现在直接运行 `train.py` 或 `inference.py` 即可。

## 📄 核心模块说明

### 1. `utils.py` - 通用工具模块
包含所有共享的工具函数和配置：
- 全局配置（设备、模型名称、超参数等）
- 数据预处理函数（文本清洗、标签构建、token对齐）
- 评测指标计算
- 模型加载函数
- 推理辅助函数
- 标点后处理规则

### 2. `training_logger.py` - 训练日志模块 ✨
**独立的日志功能模块**，与训练业务完全分离：
- `setup_training_logger()` - 日志系统初始化
- `TrainingLogCallback` - 训练进度回调类
  - 实时记录 loss、learning_rate、grad_norm 等指标
  - 自动保存到日志文件和 JSON 记录
  - 提供日志查询和摘要接口

**使用示例**:
```python
from training_logger import setup_training_logger, TrainingLogCallback

logger, log_file = setup_training_logger()
callback = TrainingLogCallback(logger)
trainer = Trainer(..., callbacks=[callback])
```

详见：[📖 日志模块使用手册](docs/TRAINING_LOGGER_MODULE.md)

### 3. `train.py` - 训练流程模块
封装完整的训练流程：
- 数据集加载和过滤
- 数据预处理和tokenization
- 类别权重计算（解决类别不平衡）
- 模型训练配置
- 训练执行（支持断点续训）
- 模型评估和保存
- **自动记录训练日志和评估结果**

### 4. `inference.py` - 推理流程模块
封装完整的推理流程：
- 模型加载
- 对测试样本进行标点预测
- 标点后处理优化
- 输出带标点的文本

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py
```

训练完成后：
- ✅ 最佳模型保存到 `punctuation_best/`
- ✅ 训练日志保存到 `logs/training_YYYYMMDD_HHMMSS.log`
- ✅ 评估结果追加到 `eval_results.json`

### 推理测试

```bash
python inference.py
```

### 查看训练日志

```bash
# 查看最新的训练日志
ls -lt logs/ | head -1

# 查看历史训练记录
cat eval_results.json
```

## ⚙️ 配置说明

在 `utils.py` 中可以修改以下配置：

```python
DEVICE = "cuda"                          # 训练设备
MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 预训练模型
DATASET_NAME = "SirlyDreamer/THUCNews"   # 数据集
MAX_SEQ_LEN = 128                        # 最大序列长度
BATCH_SIZE = 16                          # 批大小
EPOCHS = 3                               # 训练轮数
LEARNING_RATE = 2e-5                     # 学习率
```

## 📚 文档索引

### 核心文档
- [📖 训练日志模块使用手册](docs/TRAINING_LOGGER_MODULE.md) - 日志模块的完整 API 文档
- [📈 训练日志系统指南](docs/TRAINING_LOG_GUIDE.md) - 日志系统的详细说明和使用示例
- [🔍 训练进度记录说明](docs/TRAINING_PROGRESS_LOG.md) - 实时训练进度记录的实现和使用

### 技术文档
- [✨ 功能实现总结](docs/IMPLEMENTATION_SUMMARY.md) - 训练进度记录功能的实现细节
- [🔄 代码重构总结](docs/REFACTORING_SUMMARY.md) - 日志模块独立化的重构过程和优势

### 快速导航
- 💡 **新手入门**: 先阅读本 README，然后查看 [TRAINING_LOG_GUIDE.md](docs/TRAINING_LOG_GUIDE.md)
- 🔧 **开发维护**: 查看 [REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md) 了解代码结构
- 📊 **数据分析**: 参考 [TRAINING_PROGRESS_LOG.md](docs/TRAINING_PROGRESS_LOG.md) 学习如何分析训练日志

## 📊 训练日志系统

本项目实现了完整的训练日志记录功能，包括：

### 1. 实时训练进度记录

每 100 步自动记录：
- Step: 当前训练步数
- Epoch: 当前训练轮次
- Loss: 批次损失值
- Learning Rate: 当前学习率
- Grad Norm: 梯度范数

**日志示例**:
```
[训练进度] Step: 100 | Epoch: 0.0356 | Loss: 0.3807 | LR: 1.977e-05 | Grad Norm: 1.787
[训练进度] Step: 200 | Epoch: 0.0711 | Loss: 0.2945 | LR: 1.955e-05 | Grad Norm: 1.623
```

### 2. 双重保存机制

- **日志文件**: `logs/training_YYYYMMDD_HHMMSS.log` - 详细的训练过程日志
- **JSON记录**: `eval_results.json` - 结构化的训练结果汇总

### 3. 完整的参数记录

保存到 `eval_results.json` 的信息包括：
- 数据集信息（名称、样本数、划分比例）
- 超参数配置（epochs、batch_size、learning_rate等）
- 模型架构（参数量、标签映射）
- 类别权重策略和分布
- 评估指标（F1、准确率等）
- 训练环境信息（设备、时长）

详见：
- [📖 训练日志系统完整指南](docs/TRAINING_LOG_GUIDE.md)
- [📈 训练进度记录说明](docs/TRAINING_PROGRESS_LOG.md)
- [🔧 功能实现总结](docs/IMPLEMENTATION_SUMMARY.md)

## ✨ 项目特色

1. **模块化设计** - 训练、推理、日志功能完全解耦，职责清晰
2. **完整日志系统** - 实时记录训练进度，双重保存机制
3. **智能类别权重** - 自动计算类别权重，解决类别不平衡问题
4. **断点续训支持** - 支持从 checkpoint 恢复训练
5. **标点后处理** - 内置多种后处理规则，优化预测结果
6. **实验追溯** - 完整的参数记录和历史信息，便于对比分析
7. **易于扩展** - 清晰的代码结构，便于添加新功能

## 🛠️ 技术栈

- **深度学习框架**: PyTorch, Hugging Face Transformers
- **数据处理**: Datasets
- **模型**: BERT (hfl/chinese-bert-wwm-ext)
- **评估指标**: scikit-learn (F1, accuracy)
- **日志系统**: Python logging

## 📝 开发规范

### 代码组织
- 核心业务逻辑放在独立模块（`train.py`, `inference.py`）
- 通用工具函数集中在 `utils.py`
- 日志功能独立为 `training_logger.py`
- 所有文档统一放在 `docs/` 目录

### 日志规范
- 使用 `training_logger` 模块记录训练过程
- 日志文件按时间戳命名
- 评估结果自动追加到 `eval_results.json`

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 提交 Bug
- 描述清楚问题现象
- 提供复现步骤
- 附上相关日志文件

### 功能建议
- 说明需求的背景和场景
- 提供可能的实现思路
- 讨论对其他模块的影响

## 📄 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026-07-27  
**维护者**: AI Assistant
