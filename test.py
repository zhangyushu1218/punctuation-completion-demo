from data_processor import build_punct_label_sequence

# 模拟第42行
text = "约20~30度"
print(f"输入: {repr(text)}")

# 第42行：移除非法字符
text = build_punct_label_sequence(text)
print(f"输出: {repr(text)}")