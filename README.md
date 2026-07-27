# 标点符号预测项目结构说明

## 项目结构

```
asr_demo/
├── demo_punctuation.py    # 主程序入口，通过 MODE 参数控制流程
├── utils.py               # 通用工具函数和配置
├── train.py               # 训练流程
├── inference.py           # 推理流程
├── eval_results.json      # 训练评估结果记录文件（自动生成）
└── punctuation_best/      # 训练好的模型保存目录
```

## 文件说明

### 1. `utils.py` - 通用工具模块
包含所有共享的工具函数和配置：
- 全局配置（设备、模型名称、超参数等）
- 数据预处理函数（文本清洗、标签构建、token对齐）
- 评测指标计算
- 模型加载函数
- 推理辅助函数

### 2. `train.py` - 训练流程模块
封装完整的训练流程：
- 数据集加载和过滤
- 数据预处理和tokenization
- 模型训练配置
- 训练执行
- 模型评估和保存

### 3. `inference.py` - 推理流程模块
封装完整的推理流程：
- 模型加载
- 对测试样本进行标点预测
- 输出带标点的文本

### 4. `demo_punctuation.py` - 主程序入口
通过固定参数 `MODE` 控制执行哪个流程：
- `MODE = 'train'`: 执行训练流程
- `MODE = 'inference'`: 执行推理流程

### 5. `eval_results.json` - 评估结果记录文件
自动生成的 JSON 文件，记录每次训练的评估结果：
- 时间戳
- 模型名称和超参数配置
- 所有评估指标（F1、准确率等）
- 支持历史记录累积

## 使用方法

### 方式一：修改主文件中的 MODE 参数

编辑 `demo_punctuation.py` 文件，修改第 14 行：

```python
# 执行训练
MODE = 'train'

# 或执行推理
MODE = 'inference'
```

然后运行：
```bash
python demo_punctuation.py
```

### 方式二：直接运行独立模块

运行训练：
```bash
python train.py
```

运行推理：
```bash
python inference.py
```

## 配置说明

在 `utils.py` 中可以修改以下配置：

- `MODEL_NAME`: 预训练模型名称
- `RESUME_CHECKPOINT`: 已训练模型的检查点路径
- `TRAIN_CHECKPOINT`: 训练断点路径（用于续训）
- `DATASET_NAME`: 数据集名称
- `MAX_SEQ_LEN`: 最大序列长度
- `BATCH_SIZE`: 批大小
- `EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率

## 训练评估结果

每次训练完成后，系统会自动将评估结果保存到 `eval_results.json` 文件中。

### 记录内容

每个训练记录包含：
- **timestamp**: 训练完成的时间戳
- **model_name**: 使用的预训练模型名称
- **dataset**: 训练数据集名称
- **epochs**: 训练轮数
- **batch_size**: 批大小
- **learning_rate**: 学习率
- **metrics**: 所有评估指标
  - eval_loss: 验证集损失
  - eval_macro_f1: 宏平均 F1 分数
  - eval_char_accuracy: 字符级准确率
  - eval_O_f1: O 标签的 F1 分数
  - eval_COMMA_f1: 逗号预测的 F1 分数
  - eval_PERIOD_f1: 句号预测的 F1 分数
  - eval_DUN_f1: 顿号预测的 F1 分数
  - eval_runtime: 评估运行时间
  - eval_samples_per_second: 每秒处理样本数
  - eval_steps_per_second: 每秒处理步数

### 历史记录

文件采用 JSON 数组格式，支持多次训练结果的累积存储。每次训练后会自动追加新记录，方便对比不同超参数配置下的模型性能。

### 示例格式

```json
[
  {
    "timestamp": "2026-07-16 14:30:00",
    "model_name": "hfl/chinese-bert-wwm-ext",
    "dataset": "SirlyDreamer/THUCNews",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "metrics": {
      "eval_loss": 0.123456,
      "eval_macro_f1": 0.876543,
      "eval_char_accuracy": 0.945678,
      ...
    }
  }
]
```

## 优势

1. **模块化设计**：训练和推理逻辑完全解耦
2. **代码复用**：通用函数提取到 utils.py，避免重复代码
3. **易于维护**：每个模块职责单一，便于理解和修改
4. **灵活切换**：通过简单修改 MODE 参数即可切换运行模式
5. **独立运行**：各模块既可独立运行，也可通过主入口调用
