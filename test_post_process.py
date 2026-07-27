"""
测试后处理规则修复效果
"""
from utils import post_process_punctuation

def test_post_process():
    """测试后处理规则的各个场景"""
    
    test_cases = [
        {
            "name": "铁路干线-带顿号",
            "input": "并升级京、广、京、沪等干线",
            "expected": "并升级京广、京沪等干线",
        },
        {
            "name": "铁路干线-已合并",
            "input": "并升级京广京沪等干线",
            "expected": "并升级京广、京沪等干线",
        },
        {
            "name": "方位词错误",
            "input": "优化中、西部地区",
            "expected": "优化中西部地区",
        },
        {
            "name": "数量词顿号",
            "input": "增至12174列、货物列车",
            "expected": "增至12174列，货物列车",
        },
        {
            "name": "完整句子",
            "input": "7月1日起，全国铁路实行新列车运行图，图定旅客列车增至12174列、货物列车增至23975列。调图重点优化中、西部地区高铁网络，新增动车组58列，并升级京、广、京、沪等干线16列普速列车。",
            "expected_contains": ["京广、京沪"],
            "expected_not_contains": ["京、广", "中、西部", "12174列、货物"],
        }
    ]
    
    print("="*70)
    print("测试后处理规则")
    print("="*70)
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['name']}")
        print(f"输入: {case['input']}")
        
        result = post_process_punctuation(case['input'])
        print(f"输出: {result}")
        
        passed = True
        
        # 检查精确匹配
        if 'expected' in case:
            if result == case['expected']:
                print(f"✅ 精确匹配成功")
            else:
                print(f"❌ 期望: {case['expected']}")
                passed = False
        
        # 检查包含
        if 'expected_contains' in case:
            for expected in case['expected_contains']:
                if expected in result:
                    print(f"✅ 包含: {expected}")
                else:
                    print(f"❌ 应包含但未找到: {expected}")
                    passed = False
        
        # 检查不包含
        if 'expected_not_contains' in case:
            for not_expected in case['expected_not_contains']:
                if not_expected not in result:
                    print(f"✅ 正确避免: {not_expected}")
                else:
                    print(f"❌ 不应包含但找到了: {not_expected}")
                    passed = False
        
        if passed:
            print("🎉 测试通过")
        else:
            print("⚠️  测试失败")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 存在失败的测试")
    print("="*70)

if __name__ == "__main__":
    test_post_process()
