# 训练进度日志功能说明

## 📊 新增功能概述

本次更新为训练流程添加了**实时训练进度记录**功能，在每次 `logging_steps`（默认每100步）时自动记录以下关键指标：

- **Step**: 当前训练步数
- **Epoch**: 当前训练轮次（小数形式）
- **Loss**: 当前批次的损失值
- **Learning Rate**: 当前学习率
- **Grad Norm**: 梯度范数（如果可用）

---

## 🔍 日志输出示例

### 控制台和日志文件输出

```
2026-07-27 16:45:12 - INFO - 【训练阶段】开始模型训练...
2026-07-27 16:46:52 - INFO -   [训练进度] Step: 100 | Epoch: 0.0356 | Loss: 0.3807 | LR: 1.977e-05 | Grad Norm: 1.787
2026-07-27 16:48:32 - INFO -   [训练进度] Step: 200 | Epoch: 0.0711 | Loss: 0.2945 | LR: 1.955e-05 | Grad Norm: 1.623
2026-07-27 16:50:12 - INFO -   [训练进度] Step: 300 | Epoch: 0.1067 | Loss: 0.2534 | LR: 1.933e-05 | Grad Norm: 1.512
2026-07-27 16:51:52 - INFO -   [训练进度] Step: 400 | Epoch: 0.1422 | Loss: 0.2198 | LR: 1.911e-05 | Grad Norm: 1.445
...
```

### JSON 记录中的训练过程数据

在 `eval_results.json` 中，新增了 `training_process` 字段：

```json
{
  "training_process": {
    "total_steps": 8436,
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
      // ... 更多日志条目
    ],
    "logs_count": 84
  }
}
```

---

## 🎯 技术实现

### 1. 自定义 TrainerCallback

通过继承 `TrainerCallback` 类，实现了 `TrainingLogCallback` 回调函数：

```python
class TrainingLogCallback(TrainerCallback):
    """自定义回调，用于记录训练过程中的详细信息"""
    
    def __init__(self, logger):
        self.logger = logger
        self.training_logs = []  # 存储所有训练日志
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在每次 logging_steps 时触发"""
        if logs is not None:
            # 提取关键信息并记录
            log_entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 4),
                "loss": round(logs.get("loss", 0), 4),
                "learning_rate": logs.get("learning_rate", 0),
            }
            
            # 如果有梯度范数则记录
            if "grad_norm" in logs:
                log_entry["grad_norm"] = round(logs["grad_norm"], 4)
            
            # 保存到列表
            self.training_logs.append(log_entry)
            
            # 格式化输出到日志文件
            log_message = f"Step: {log_entry['step']} | Epoch: {log_entry['epoch']} | ..."
            self.logger.info(f"  [训练进度] {log_message}")
```

### 2. 注册回调到 Trainer

```python
# 创建训练日志回调
training_log_callback = TrainingLogCallback(logger)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[training_log_callback]  # 注册回调
)
```

---

## 💡 使用优势

### 1. **实时监控训练状态**
- 每100步自动输出一次训练进度
- 可以观察 loss 的下降趋势
- 监控学习率的变化（如果使用学习率调度器）

### 2. **问题诊断**
- **Loss 不下降**: 可能学习率过大或过小
- **Grad Norm 异常**: 梯度爆炸或消失的迹象
- **Loss 波动大**: 可能需要调整 batch size 或学习率

### 3. **性能分析**
- 通过 epoch 和 step 的关系了解训练速度
- 分析不同阶段的收敛情况
- 识别训练瓶颈

### 4. **实验对比**
- 保存的训练日志可用于对比不同超参数的效果
- 可以绘制 loss 曲线图进行可视化分析
- 便于复现和调试

---

## 📈 典型应用场景

### 场景1: 监控训练收敛

通过观察日志中的 loss 变化：
```
Step: 100  | Loss: 0.3807  ← 初始阶段，loss较高
Step: 500  | Loss: 0.1987  ← 快速下降
Step: 1000 | Loss: 0.1428  ← 逐渐收敛
Step: 1500 | Loss: 0.1200  ← 趋于稳定
```

### 场景2: 检测梯度问题

如果 grad_norm 异常：
```
Step: 100  | Grad Norm: 1.787   ← 正常
Step: 200  | Grad Norm: 15.234  ← ⚠️ 梯度爆炸！
Step: 300  | Grad Norm: 0.001   ← ⚠️ 梯度消失！
```

### 场景3: 验证学习率调度

观察 learning_rate 的变化：
```
Step: 100  | LR: 1.977e-05  ← 预热阶段
Step: 500  | LR: 2.000e-05  ← 达到峰值
Step: 1000 | LR: 1.778e-05  ← 开始衰减
Step: 1500 | LR: 1.333e-05  ← 持续衰减
```

---

## 🔧 配置说明

### 调整日志频率

在 `TrainingArguments` 中修改 `logging_steps` 参数：

```python
training_args = TrainingArguments(
    output_dir="../punctuation_model",
    logging_steps=50,  # 改为每50步记录一次（更频繁）
    # logging_steps=200,  # 或每200步记录一次（更少）
    ...
)
```

### 完整保存所有训练日志

如果需要保存所有训练步骤的详细日志（而非仅前10条），可以修改代码：

```python
"training_process": {
    "total_steps": len(training_log_callback.training_logs),
    "logs_all": training_log_callback.training_logs,  # 保存全部
    "logs_count": len(training_log_callback.training_logs)
}
```

⚠️ **注意**: 完整保存会显著增加 JSON 文件大小，建议仅在需要详细分析时使用。

---

## 📊 数据分析示例

### 绘制 Loss 曲线

可以使用以下 Python 代码读取并可视化训练日志：

```python
import json
import matplotlib.pyplot as plt

# 读取训练记录
with open('eval_results.json', 'r') as f:
    records = json.load(f)

# 获取最新的训练记录
latest = records[-1]
logs = latest['training_process']['logs_sample']

# 提取数据
steps = [log['step'] for log in logs]
losses = [log['loss'] for log in logs]
lrs = [log['learning_rate'] for log in logs]

# 绘制 Loss 曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, 'b-', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, lrs, 'r-', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

## ✨ 总结

通过本次更新，训练流程现在能够：

✅ **实时记录** - 每100步自动记录训练进度  
✅ **详细指标** - 包含 loss、lr、grad_norm 等关键指标  
✅ **双重保存** - 同时保存到日志文件和 JSON 记录  
✅ **便于分析** - 结构化数据支持可视化和对比分析  
✅ **问题诊断** - 帮助快速定位训练异常  

这为模型训练提供了完整的可观测性，让你能够更好地理解和优化训练过程！🚀
