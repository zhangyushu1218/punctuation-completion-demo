from utils import (
    infer_punctuation,
    load_model_and_tokenizer, post_process_punctuation
)


def inference_model():
    """执行推理流程"""
    print("=" * 50)
    print("开始推理流程")
    print("=" * 50)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()

    # 测试样本
    test_asr_texts = [
        "今天早上出门去公园散步看到很多老人打太极还有小朋友放风筝",
        "机器学习深度学习大模型自然语言处理计算机视觉多模态技术快速发展",
        "苹果香蕉橘子葡萄西瓜都是夏季常见水果口感清甜汁水丰富",
        "7月1日起全国铁路实行新列车运行图图定旅客列车增至12174列货物列车增至23975列调图重点优化中西部地区高铁网络新增动车组58列并升级京广京沪等干线16列普速列车"
    ]

    print("\n===== ASR无标点文本自动标点推理示例 =====")
    for text in test_asr_texts:
        res = infer_punctuation(text, model, tokenizer)
        # 后处理：修正常见的并列专有名词模式
        # post_res = post_process_punctuation(res)
        print(f"原始无标点：{text}")
        print(f"模型输出：{res}")
        # print(f"后处理输出：{post_res}\n")


if __name__ == "__main__":
    inference_model()
