
def post_process_punctuation(text: str) -> str:
    """
    后处理标点符号，修复常见模式

    规则：
    1. 修正数字后的顿号（12174列、货物 -> 12174列，货物）- 优先级最高
    2. 修复铁路干线名称（京、广、京、沪 -> 京广、京沪）
    3. 移除单字后的顿号（如：中、西部 -> 中西部）
    4. 确保双引号成对出现（左引号和右引号匹配）
    """
    import re

    # 规则1: 先修正"数量词 + 顿号 + 名词"的错误（应该是逗号）- 优先级最高
    # 例如："12174列、货物" -> "12174列，货物"
    # 必须在其他规则之前执行，避免顿号被提前移除
    number_dunhao = r'(\d+[\u4e00-\u9fa5]*)、([\u4e00-\u9fa5])'
    text = re.sub(number_dunhao, r'\1，\2', text)

    # 规则2: 处理铁路干线名称模式
    # 模式1: 带顿号的省份简称序列 "京、广、京、沪" -> "京广、京沪"
    railway_with_dunhao = r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])、([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])'

    def replace_railway_with_dunhao(match):
        """将'京、广、京、沪'转换为'京广、京沪'"""
        return f"{match.group(1)}{match.group(2)}、{match.group(3)}{match.group(4)}"

    text = re.sub(railway_with_dunhao, replace_railway_with_dunhao, text)

    # 模式2: 已经合并的铁路干线名称 "京广京沪" -> "京广、京沪"
    # 使用边界确保只匹配4个连续的省份简称字符
    railway_pattern = r'(?<![京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{2})([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]{2})(?![京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼])'

    def replace_railway(match):
        """将'京广京沪'转换为'京广、京沪'"""
        return f"{match.group(1)}、{match.group(2)}"

    text = re.sub(railway_pattern, replace_railway, text)

    # 规则3: 移除单字之间的顿号（过度预测的顿号）
    # 匹配模式：单字 + 顿号 + 单字，如"中、西"、"东、西"
    single_char_dunhao = r'([\u4e00-\u9fa5])、([\u4e00-\u9fa5])(?=[\u4e00-\u9fa5]|$)'

    def fix_single_char_dunhao(match):
        """移除单字之间的顿号"""
        char1 = match.group(1)
        char2 = match.group(2)

        # 检查是否是常见的单字省份简称组合（应该合并）
        province_chars = set('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼')
        if char1 in province_chars and char2 in province_chars:
            # 这是省份简称，应该合并（如"京广"、"京沪"）
            return f"{char1}{char2}"

        # 其他单字组合也默认合并（避免"中、西部"这样的错误）
        return f"{char1}{char2}"

    text = re.sub(single_char_dunhao, fix_single_char_dunhao, text)

    # 规则4: 确保双引号成对出现
    # 统计左引号和右引号的数量，如果不匹配则进行调整
    left_quotes = text.count('“')
    right_quotes = text.count('”')

    if left_quotes > right_quotes:
        # 左引号多于右引号，在末尾补充右引号（如果文本以非引号结尾）
        diff = left_quotes - right_quotes
        if not text.endswith('”'):
            text += '”' * diff
    elif right_quotes > left_quotes:
        # 右引号多于左引号，在开头补充左引号
        diff = right_quotes - left_quotes
        if not text.startswith('“'):
            text = '“' * diff + text

    return text