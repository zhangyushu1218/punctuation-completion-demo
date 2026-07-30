# 标点符号扩展说明

## 概述
本次更新将模型的标点符号支持从3种扩展到10种，增强了模型对中文文本的理解和标注能力。

## 支持的标点符号

### 原有标点（3种）
1. **逗号** (，) - COMMA
2. **句号** (。) - PERIOD
3. **顿号** (、) - DUN

### 新增标点（7种）
4. **感叹号** (！) - EXCLAMATION
5. **问号** (？) - QUESTION
6. **冒号** (：) - COLON
7. **分号** (；) - SEMICOLON
8. **左双引号** (") - QUOTE_LEFT
9. **右双引号** (") - QUOTE_RIGHT

## 技术实现

### 1. 标签配置 (utils.py)
```python
LABELS = [
    "O",              # 0: 无标点
    "COMMA",          # 1: 逗号
    "PERIOD",         # 2: 句号
    "DUN",            # 3: 顿号
    "EXCLAMATION",    # 4: 感叹号
    "QUESTION",       # 5: 问号
    "COLON",          # 6: 冒号
    "SEMICOLON",      # 7: 分号
    "QUOTE_LEFT",     # 8: 左双引号
    "QUOTE_RIGHT"     # 9: 右双引号
]
NUM_LABELS = 10
```

### 2. 标点映射
所有标点均使用中文全角字符：
- 感叹号：`！` (U+FF01)
- 问号：`？` (U+FF1F)
- 冒号：`：` (U+FF1A)
- 分号：`；` (U+FF1B)
- 左双引号：`"` (U+201C)
- 右双引号：`"` (U+201D)

### 3. 数据清洗
更新了 `clean_raw_text()` 函数的正则表达式，保留所有新标点：
```python
text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。、！？：；\u201c\u201d]", "", text)
```

### 4. 后处理增强
在 `post_process_punctuation()` 中新增规则4：确保双引号成对出现
- 如果左引号多于右引号，在末尾补充右引号
- 如果右引号多于左引号，在开头补充左引号

## 影响范围

### 需要重新训练
由于标签数量从4个增加到10个，**必须重新训练模型**才能使用新的标点符号功能。

### 兼容性
- ✅ 现有代码无需修改（train.py, inference.py, main.py）
- ✅ 所有配置集中在 utils.py，便于维护
- ✅ 评估指标自动适配新标签（compute_metrics 函数）

### 文件大小
- 模型输出层从 4 类变为 10 类
- 参数量略有增加（分类层权重矩阵变化）
- 总参数量仍远低于 150M 限制

## 使用示例

### 训练
```bash
python main.py  # MODE = 'train'
```

### 推理
```python
from utils import infer_punctuation, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()

# 测试包含新标点的文本
text = "你真的要去吗他问当然我要去她回答"
result = infer_punctuation(text, model, tokenizer)
# 预期输出: "你真的要去吗？他问。当然，我要去。她回答。"
```

## 注意事项

1. **数据集要求**: 训练数据需要包含足够的新标点样本，特别是：
   - 感叹句和疑问句
   - 冒号和分号的使用场景
   - 引用对话的文本

2. **类别平衡**: 新标点的出现频率可能不均衡，建议：
   - 检查各类别分布
   - 必要时调整类别权重策略

3. **评估指标**: 每个标点都有独立的 F1 分数，可以单独评估各标点的预测效果

## 测试验证
运行测试脚本验证配置正确性：
```bash
python test_extended_punctuation.py
```

## 版本信息
- 更新日期: 2026-07-30
- 修改文件: utils.py
- 标签数量: 4 → 10
- 标点类型: 3 → 9 (+ 双引号左右)
