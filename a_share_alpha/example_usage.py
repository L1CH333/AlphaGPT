"""
示例脚本：A股因子挖掘

演示如何使用 A-Share AlphaGPT 进行因子挖掘
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a_share_alpha.engine import AShareAlphaEngine
from a_share_alpha.config import AShareConfig


def example_1_quick_start():
    """示例 1: 快速开始 - 使用模拟数据"""
    print("=" * 60)
    print("示例 1: 使用模拟数据快速测试")
    print("=" * 60)
    
    # 创建引擎（使用模拟数据）
    engine = AShareAlphaEngine(use_simulated_data=True)
    
    # 训练 500 步（快速测试）
    engine.train(steps=500)
    
    # 评估最佳策略
    engine.evaluate_best_strategy()


def example_2_custom_config():
    """示例 2: 自定义配置参数"""
    print("=" * 60)
    print("示例 2: 自定义配置参数")
    print("=" * 60)
    
    # 修改配置
    AShareConfig.BATCH_SIZE = 2048
    AShareConfig.TRAIN_STEPS = 1000
    AShareConfig.TRADE_SIZE_CNY = 50000  # 5万元
    
    # 创建引擎
    engine = AShareAlphaEngine(use_simulated_data=True)
    
    # 训练
    engine.train()
    
    # 评估
    engine.evaluate_best_strategy()


def example_3_database_connection():
    """示例 3: 连接真实数据库"""
    print("=" * 60)
    print("示例 3: 连接数据库加载真实 Tick 数据")
    print("=" * 60)
    
    # 设置数据库连接（确保环境变量已设置）
    # export DB_USER=your_user
    # export DB_PASSWORD=your_password
    # export DB_HOST=localhost
    # export DB_NAME=a_share_tick
    
    try:
        # 创建引擎（使用真实数据）
        engine = AShareAlphaEngine(use_simulated_data=False)
        
        # 训练
        engine.train(steps=2000)
        
        # 评估
        engine.evaluate_best_strategy()
        
    except Exception as e:
        print(f"数据库连接失败: {e}")
        print("请确保数据库连接配置正确，或使用模拟数据模式")


def example_4_manual_factor_test():
    """示例 4: 手动测试特定因子"""
    print("=" * 60)
    print("示例 4: 手动测试特定因子公式")
    print("=" * 60)
    
    from a_share_alpha.data_loader import AShareTickDataLoader
    from a_share_alpha.vm import AShareStackVM
    from a_share_alpha.backtest import AShareBacktest
    
    # 加载数据
    loader = AShareTickDataLoader()
    loader._load_simulated_data(n_stocks=50, n_ticks=200)
    
    # 创建 VM 和回测引擎
    vm = AShareStackVM()
    bt = AShareBacktest()
    
    # 手动定义因子公式
    # Stack-based postfix notation:
    # To compute: F0 - DELAY5(F0), we need: F0 F0 DELAY5 SUB
    # 特征索引 0-9, 操作符从 10 开始
    # F0=0, DELAY5=20, SUB=11
    
    formula = [0, 0, 20, 11]  # F0 F0 DELAY5 SUB (momentum factor)
    
    print(f"测试因子: F0 - DELAY5(F0) (动量因子)")
    
    # 执行因子
    factor_values = vm.execute(formula, loader.feat_tensor)
    
    if factor_values is not None:
        # 回测
        metrics = bt.detailed_backtest(
            factor_values,
            loader.raw_data_cache,
            loader.target_ret
        )
        
        print(f"\n回测结果:")
        print(f"  总收益: {metrics['total_return']:.4%}")
        print(f"  夏普比率: {metrics['sharpe']:.4f}")
        print(f"  胜率: {metrics['win_rate']:.4%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.4%}")
    else:
        print("因子公式无效")


def example_5_batch_test():
    """示例 5: 批量测试多个因子"""
    print("=" * 60)
    print("示例 5: 批量测试预定义因子库")
    print("=" * 60)
    
    from a_share_alpha.data_loader import AShareTickDataLoader
    from a_share_alpha.vm import AShareStackVM
    from a_share_alpha.backtest import AShareBacktest
    
    # 加载数据
    loader = AShareTickDataLoader()
    loader._load_simulated_data(n_stocks=50, n_ticks=200)
    
    vm = AShareStackVM()
    bt = AShareBacktest()
    
    # 预定义因子库 (使用栈式后缀表达式)
    factor_library = {
        "动量因子": [0, 0, 20, 11],       # F0 - DELAY5(F0)
        "订单流×成交量": [1, 5, 12],      # F1 * F5
        "价差标准化": [2, 26],            # STD10(F2)
        "反转因子": [0, 19, 14],          # NEG(DELAY1(F0))
    }
    
    results = []
    
    print("\n开始批量测试...\n")
    
    for name, formula in factor_library.items():
        factor_values = vm.execute(formula, loader.feat_tensor)
        
        if factor_values is not None:
            score, ret = bt.evaluate(
                factor_values,
                loader.raw_data_cache,
                loader.target_ret
            )
            
            results.append({
                'name': name,
                'score': score.item(),
                'return': ret
            })
            
            print(f"{name:15s} | Score: {score:.4f} | Return: {ret:.4%}")
        else:
            print(f"{name:15s} | 无效公式")
    
    # 排序并显示最佳因子
    if results:
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"\n最佳因子: {results[0]['name']}")
        print(f"  评分: {results[0]['score']:.4f}")
        print(f"  收益: {results[0]['return']:.4%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A-Share AlphaGPT 示例脚本")
    parser.add_argument(
        '--example',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='选择要运行的示例 (1-5)'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_quick_start,
        2: example_2_custom_config,
        3: example_3_database_connection,
        4: example_4_manual_factor_test,
        5: example_5_batch_test
    }
    
    print("\n" + "=" * 60)
    print("A-Share AlphaGPT 因子挖掘示例")
    print("=" * 60 + "\n")
    
    examples[args.example]()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60 + "\n")
