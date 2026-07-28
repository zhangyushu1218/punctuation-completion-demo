# 训练日志模块使用说明

## 📦 模块概述

`training_logger.py` 是一个独立的训练日志模块，提供了完整的训练过程日志记录和进度追踪功能。该模块与训练业务逻辑完全分离，可以在任何使用 Hugging Face Transformers Trainer 的项目中复用。

---

## 🏗️ 模块结构

```
training_logger.py
├── setup_training_logger()     # 日志系统初始化函数
└── TrainingLogCallback         # 训练进度回调类
    ├── __init__()              # 初始化
    ├── on_log()                # 日志记录回调
    ├── get_training_logs()     # 获取完整日志
    └── get_logs_summary()      # 获取日志摘要
```

---

## 🚀 快速开始

### 1. 基本用法

```python
from training_logger import setup_training_logger, TrainingLogCallback

# 初始化日志系统
logger, log_file = setup_training_logger()

# 创建训练日志回调
callback = TrainingLogCallback(logger)

# 在 Trainer 中使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[callback]  # 注册回调
)

# 开始训练
trainer.train()

# 获取训练日志
logs = callback.get_training_logs()
summary = callback.get_logs_summary()
```

### 2. 自定义日志目录

```python
# 指定自定义日志目录
logger, log_file = setup_training_logger(log_dir="./my_logs")
```

---

## 📋 API 详解

### `setup_training_logger(log_dir="./logs")`

**功能**: 初始化训练日志系统，创建带时间戳的日志文件

**参数**:
- `log_dir` (str): 日志文件保存目录，默认为 `"./logs"`

**返回**:
- `tuple`: `(logger, log_file_path)`
  - `logger`: logging.Logger 实例
  - `log_file_path`: 日志文件完整路径

**特性**:
- ✅ 自动创建日志目录
- ✅ 使用时间戳命名（格式：`training_YYYYMMDD_HHMMSS.log`）
- ✅ 双输出模式（文件 + 控制台）
- ✅ UTF-8 编码支持中文
- ✅ 自动清除已有 handlers

**示例**:
```python
logger, log_file = setup_training_logger()
logger.info("训练开始")
# 输出到控制台和控制台
# 同时保存到 ./logs/training_20260727_164500.log
```

---

### `TrainingLogCallback(logger)`

**功能**: 自定义 Trainer 回调，记录训练过程中的详细指标

**参数**:
- `logger`: logging.Logger 实例

**记录的指标**:
- `step`: 当前训练步数
- `epoch`: 当前训练轮次（小数形式）
- `loss`: 批次损失值
- `learning_rate`: 当前学习率
- `grad_norm`: 梯度范数（如果可用）

**方法**:

#### `on_log(args, state, control, logs=None, **kwargs)`

在每次 `logging_steps` 时自动触发，记录训练进度。

**参数**:
- `args`: TrainingArguments
- `state`: TrainerState
- `control`: TrainerControl
- `logs`: 包含训练指标的字典

**示例输出**:
```
[训练进度] Step: 100 | Epoch: 0.0356 | Loss: 0.3807 | LR: 1.977e-05 | Grad Norm: 1.787
```

#### `get_training_logs()`

获取所有记录的训练日志。

**返回**:
- `list`: 训练日志列表，每个元素是一个字典

**示例**:
```python
logs = callback.get_training_logs()
print(logs[0])
# {
#     "step": 100,
#     "epoch": 0.0356,
#     "loss": 0.3807,
#     "learning_rate": 1.977e-05,
#     "grad_norm": 1.787
# }
```

#### `get_logs_summary()`

获取训练日志摘要（用于保存到 JSON）。

**返回**:
- `dict`: 包含日志统计信息的字典
  ```python
  {
      "total_steps": 84,
      "logs_sample": [...],  # 前10条日志
      "logs_count": 84
  }
  ```

---

## 💡 使用场景

### 场景1: 标准训练流程

```python
from training_logger import setup_training_logger, TrainingLogCallback
from transformers import Trainer, TrainingArguments

# 初始化日志
logger, log_file = setup_training_logger()

# 创建回调
callback = TrainingLogCallback(logger)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./model",
    logging_steps=100,  # 每100步记录一次
    ...
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[callback]
)

# 训练
trainer.train()

# 获取日志摘要
summary = callback.get_logs_summary()
```

### 场景2: 多实验对比

```python
experiments = [
    {"lr": 1e-5, "batch_size": 16},
    {"lr": 2e-5, "batch_size": 16},
    {"lr": 2e-5, "batch_size": 32},
]

results = []

for i, config in enumerate(experiments):
    logger, log_file = setup_training_logger()
    callback = TrainingLogCallback(logger)
    
    logger.info(f"实验 {i+1}: {config}")
    
    # 训练...
    trainer = Trainer(..., callbacks=[callback])
    trainer.train()
    
    results.append({
        "config": config,
        "logs_summary": callback.get_logs_summary()
    })
```

### 场景3: 自定义日志分析

```python
callback = TrainingLogCallback(logger)
trainer = Trainer(..., callbacks=[callback])
trainer.train()

# 获取完整日志
logs = callback.get_training_logs()

# 分析 loss 变化
import matplotlib.pyplot as plt

steps = [log['step'] for log in logs]
losses = [log['loss'] for log in logs]

plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
plt.show()
```

---

## 🔧 高级配置

### 调整日志频率

在 `TrainingArguments` 中设置 `logging_steps`:

```python
training_args = TrainingArguments(
    logging_steps=50,   # 更频繁：每50步
    # logging_steps=200,  # 更少：每200步
    ...
)
```

### 自定义日志格式

修改 `setup_training_logger` 中的 formatter:

```python
file_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

### 添加额外日志级别

```python
logger.setLevel(logging.DEBUG)  # 更详细的日志
file_handler.setLevel(logging.DEBUG)
```

---

## 📊 日志输出示例

### 控制台输出

```
================================================================================
开始训练流程
================================================================================

【数据准备】加载THUCNews数据集...
训练集样本数: 45000
验证集样本数: 5000

【模型加载】加载模型和分词器...
模型总参数量: 102.27M (102,267,652)

【训练阶段】开始模型训练...
  [训练进度] Step: 100 | Epoch: 0.0356 | Loss: 0.3807 | LR: 1.977e-05 | Grad Norm: 1.787
  [训练进度] Step: 200 | Epoch: 0.0711 | Loss: 0.2945 | LR: 1.955e-05 | Grad Norm: 1.623
  [训练进度] Step: 300 | Epoch: 0.1067 | Loss: 0.2534 | LR: 1.933e-05 | Grad Norm: 1.512
  ...

训练完成！总耗时: 1830.25 秒 (30.50 分钟)
```

### 日志文件内容

```
2026-07-27 16:45:00 - INFO - ================================================================================
2026-07-27 16:45:00 - INFO - 开始训练流程
2026-07-27 16:45:00 - INFO - ================================================================================
2026-07-27 16:45:05 - INFO - 训练集样本数: 45000
2026-07-27 16:45:05 - INFO - 验证集样本数: 5000
2026-07-27 16:46:52 - INFO -   [训练进度] Step: 100 | Epoch: 0.0356 | Loss: 0.3807 | LR: 1.977e-05 | Grad Norm: 1.787
2026-07-27 16:48:32 - INFO -   [训练进度] Step: 200 | Epoch: 0.0711 | Loss: 0.2945 | LR: 1.955e-05 | Grad Norm: 1.623
...
```

### JSON 记录

```json
{
  "training_process": {
    "total_steps": 84,
    "logs_sample": [
      {
        "step": 100,
        "epoch": 0.0356,
        "loss": 0.3807,
        "learning_rate": 1.977e-05,
        "grad_norm": 1.787
      },
      {
        "step": 200,
        "epoch": 0.0711,
        "loss": 0.2945,
        "learning_rate": 1.955e-05,
        "grad_norm": 1.623
      }
    ],
    "logs_count": 84
  }
}
```

---

## 🎯 最佳实践

### 1. 日志目录管理

定期清理旧日志文件，避免占用过多磁盘空间：

```python
import os
import glob

# 删除7天前的日志
log_files = glob.glob("./logs/training_*.log")
for log_file in log_files:
    if os.path.getmtime(log_file) < time.time() - 7 * 86400:
        os.remove(log_file)
```

### 2. 日志级别控制

生产环境使用 INFO，调试时使用 DEBUG：

```python
import sys

if "--debug" in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
```

### 3. 异常处理

确保日志系统在异常情况下也能正常工作：

```python
try:
    logger, log_file = setup_training_logger()
    callback = TrainingLogCallback(logger)
    # 训练代码...
except Exception as e:
    logger.error(f"训练失败: {e}", exc_info=True)
    raise
```

### 4. 性能优化

对于长时间训练，可以只保存摘要而非全部日志：

```python
# 只保存摘要，节省存储空间
summary = callback.get_logs_summary()
```

---

## 🔍 故障排查

### 问题1: 日志文件未生成

**原因**: 目录权限问题或路径错误

**解决**:

```python
# 检查目录是否存在且可写
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
print(f"日志目录: {os.path.abspath(log_dir)}")
```

### 问题2: 控制台无输出

**原因**: Handler 配置问题

**解决**:
```python
# 检查 handlers
print(f"Handlers: {logger.handlers}")
print(f"Level: {logger.level}")
```

### 问题3: 中文乱码

**原因**: 编码设置问题

**解决**:
```python
# 确保使用 UTF-8 编码
file_handler = logging.FileHandler(log_file, encoding='utf-8')
```

---

## 📝 模块优势

✅ **独立性**: 与训练业务完全分离，可独立测试和维护  
✅ **复用性**: 可在任何 Transformers 项目中直接使用  
✅ **灵活性**: 支持自定义日志目录、格式和级别  
✅ **完整性**: 提供完整的日志记录和查询接口  
✅ **易用性**: 简单的 API，开箱即用  

---

## 🔄 版本历史

### v1.0.0 (2026-07-27)
- ✅ 初始版本发布
- ✅ 实现 `setup_training_logger()` 函数
- ✅ 实现 `TrainingLogCallback` 类
- ✅ 支持双输出模式（文件 + 控制台）
- ✅ 支持训练进度实时记录
- ✅ 提供日志查询和摘要接口

---

## 📞 支持与反馈

如有问题或建议，请查看项目文档或提交 Issue。
