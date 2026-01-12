# A-Share Factor Mining with AlphaGPT

## 概述 (Overview)

本项目基于 AlphaGPT 框架，专门适配中国 A 股市场的因子挖掘，使用股票逐笔成交数据（Tick Data）进行高频因子发现。

This project adapts the AlphaGPT framework specifically for Chinese A-share market factor mining, using tick-by-tick transaction data for high-frequency factor discovery.

## 核心特性 (Key Features)

### 1. 逐笔数据处理 (Tick Data Processing)
- 支持逐笔成交数据加载和预处理
- 10 种 tick 级别的市场微观结构特征
- 订单流不平衡、买卖价差、VWAP 偏离等指标

### 2. A 股市场特性适配 (A-Share Market Adaptation)
- **交易规则**: T+1 交易制度、涨跌停限制（±10%，ST 股票 ±5%）
- **交易成本**: 佣金（0.03%）、印花税（0.1% 仅卖出）、最低佣金（5元）
- **市场特征**: 成交量阈值、流动性过滤

### 3. 符号回归因子挖掘 (Symbolic Factor Mining)
- 基于 Transformer 的自回归因子生成
- 27 种时序操作算子（延迟、差分、滚动统计等）
- 强化学习优化因子收益

### 4. 回测引擎 (Backtesting Engine)
- 考虑 A 股交易成本的完整回测
- 风险调整后的收益评估
- Sharpe 比率、胜率、最大回撤等指标

## 项目结构 (Project Structure)

```
a_share_alpha/
├── __init__.py          # 模块初始化
├── config.py            # A 股市场配置参数
├── tick_features.py     # Tick 数据特征工程
├── data_loader.py       # 逐笔数据加载器
├── ops.py              # 时序操作算子库
├── vm.py               # 因子执行虚拟机
├── backtest.py         # A 股回测引擎
├── alphagpt.py         # AlphaGPT 模型
├── engine.py           # 主训练引擎
└── README.md           # 本文档
```

## 快速开始 (Quick Start)

### 1. 环境要求 (Requirements)

```bash
Python >= 3.8
PyTorch >= 2.0
pandas >= 1.5
sqlalchemy >= 2.0
tqdm
```

### 2. 数据准备 (Data Preparation)

#### 方式 1: 使用模拟数据（快速测试）
```python
from a_share_alpha.engine import AShareAlphaEngine

# 使用模拟数据
engine = AShareAlphaEngine(use_simulated_data=True)
engine.train(steps=500)
engine.evaluate_best_strategy()
```

#### 方式 2: 连接真实数据库
设置环境变量：
```bash
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_NAME=a_share_tick
```

数据库表结构：
```sql
CREATE TABLE tick_data (
    timestamp TIMESTAMP,
    stock_code VARCHAR(10),
    price DECIMAL(10, 2),
    volume BIGINT,
    buy_volume BIGINT,
    sell_volume BIGINT,
    bid_price_1 DECIMAL(10, 2),
    ask_price_1 DECIMAL(10, 2),
    bid_volume_1 BIGINT,
    ask_volume_1 BIGINT
);
```

### 3. 运行因子挖掘 (Run Factor Mining)

```python
from a_share_alpha.engine import AShareAlphaEngine

# 初始化引擎
engine = AShareAlphaEngine(use_simulated_data=False)

# 训练模型发现因子
engine.train(steps=2000)

# 评估最佳策略
engine.evaluate_best_strategy()
```

## 特征说明 (Feature Description)

系统自动从逐笔数据计算 10 个特征：

| 特征编号 | 名称 | 说明 |
|---------|------|------|
| F0 | Tick Return | 逐笔收益率 |
| F1 | Order Flow Imbalance | 订单流不平衡度 |
| F2 | Bid-Ask Spread | 买卖价差 |
| F3 | VWAP Deviation | 成交量加权价格偏离 |
| F4 | Price Momentum | 价格动量 |
| F5 | Volume Surge | 成交量激增 |
| F6 | Tick Direction Strength | Tick 方向强度 |
| F7 | Large Order Ratio | 大单占比 |
| F8 | Log Volume | 对数成交量 |
| F9 | Volume Ratio | 成交量比率 |

## 操作算子 (Operators)

系统提供 27 种操作算子用于构建因子：

### 基础运算 (Arithmetic)
- ADD, SUB, MUL, DIV
- NEG, ABS, SIGN, SQUARE, SQRT

### 时序操作 (Time Series)
- DELAY1, DELAY5: 时间延迟
- DELTA1, DELTA5: 差分
- MA5, MA10, MA20: 移动平均
- STD10, STD20: 滚动标准差
- MAX5, MAX10, MIN5, MIN10: 滚动极值
- RANK10, RANK20: 滚动排名

### 高级操作 (Advanced)
- GATE: 条件选择
- DECAY: 指数衰减
- CORR: 滚动相关系数

## 示例因子 (Example Factors)

```python
# 示例 1: 动量因子
# F0 SUB DELAY5 -> (当前收益率 - 5期前收益率)

# 示例 2: 订单流因子
# F1 MUL F5 -> (订单流不平衡 × 成交量激增)

# 示例 3: 复合因子
# F0 MA10 F3 SUB RANK20 -> RANK((MA10(return) - VWAP_dev), 20)
```

## 配置参数 (Configuration)

在 `config.py` 中可以调整以下参数：

```python
# 训练参数
BATCH_SIZE = 4096           # 批次大小
TRAIN_STEPS = 2000          # 训练步数
MAX_FORMULA_LEN = 15        # 因子公式最大长度

# 交易参数
TRADE_SIZE_CNY = 100000     # 交易金额（元）
COMMISSION_RATE = 0.0003    # 佣金率
STAMP_TAX = 0.001          # 印花税
```

## 输出结果 (Output)

训练完成后，最佳策略保存在 `output/best_ashare_strategy.json`：

```json
{
  "score": 15.234,
  "formula_tokens": [0, 1, 5, ...],
  "formula_readable": "F0 F1 MUL F5 ADD",
  "config": {
    "max_formula_len": 15,
    "input_dim": 10
  }
}
```

## 评估指标 (Evaluation Metrics)

- **Total Return**: 累计收益率
- **Sharpe Ratio**: 夏普比率
- **Win Rate**: 胜率
- **Max Drawdown**: 最大回撤
- **Avg Trades**: 平均交易次数

## 与原版对比 (Comparison with Original)

| 特性 | 原版 (Crypto) | A股版本 (A-Share) |
|-----|--------------|------------------|
| 数据类型 | OHLCV | Tick-by-tick |
| 特征数量 | 6 | 10 |
| 操作算子 | 12 | 27 |
| 交易规则 | 24/7, 0.5% fee | T+1, 佣金+印花税 |
| 市场特性 | 高波动 Meme 币 | A股监管市场 |

## 注意事项 (Notes)

1. **数据质量**: Tick 数据质量对因子效果影响很大，建议使用高质量的 Level-2 行情数据
2. **过拟合**: 建议使用滚动窗口进行样本外测试
3. **实盘差异**: 回测结果不代表实盘表现，需考虑滑点、冲击成本等
4. **合规性**: 请确保数据来源和使用符合相关法规

## 未来改进 (Future Improvements)

- [ ] 支持分钟 K 线数据
- [ ] 增加截面因子（横截面排名）
- [ ] 多因子组合优化
- [ ] 因子 IC/IR 分析
- [ ] 支持更多交易所（深交所、科创板等）

## 许可证 (License)

本项目继承 AlphaGPT 的开源许可证。

## 参考 (References)

- AlphaGPT 原始项目
- WorldQuant 101 Alphas
- A-Share 市场微观结构研究

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。使用本项目进行实盘交易的风险由使用者自行承担。
