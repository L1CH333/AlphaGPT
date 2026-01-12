# AlphaGPT4as

**AlphaGPT for A-Share Market** - 基于深度强化学习的 A 股因子挖掘系统

A deep reinforcement learning-based factor mining system for Chinese A-share market using tick-by-tick transaction data.

---

## 🎯 项目简介 (Overview)

AlphaGPT4as 是专门为中国 A 股市场设计的因子挖掘系统，基于 AlphaGPT 框架，使用逐笔成交数据（Tick Data）进行高频因子发现。

AlphaGPT4as is a factor mining system specifically designed for the Chinese A-share market, based on the AlphaGPT framework, using tick-by-tick transaction data for high-frequency factor discovery.

### 核心特性 (Key Features)

✅ **逐笔数据处理** - 支持 tick 级别的高频交易数据  
✅ **A 股市场适配** - T+1 交易制度、涨跌停限制、真实交易成本  
✅ **10 种微观结构特征** - 订单流、买卖价差、VWAP 偏离等  
✅ **27 种时序操作算子** - 延迟、滚动统计、排名等专业算子  
✅ **强化学习优化** - 基于 Transformer 的自动因子生成  
✅ **完整回测引擎** - 考虑真实交易成本的性能评估  

---

## 📦 项目结构 (Project Structure)

```
AlphaGPT4as/
├── README.md              # 本文档
├── DETAILED_DOC.md        # 详细文档和使用指南
├── IMPLEMENTATION.md      # 实现总结和技术细节
├── __init__.py           # 模块初始化
├── config.py             # 配置参数
├── tick_features.py      # Tick 特征工程（10 种特征）
├── data_loader.py        # 数据加载器
├── ops.py               # 时序操作算子（27 种）
├── vm.py                # 因子执行引擎
├── backtest.py          # A 股回测引擎
├── alphagpt.py          # AlphaGPT 模型
├── engine.py            # 主训练引擎
└── example_usage.py     # 使用示例（5 个）
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境要求 (Requirements)

```bash
pip install torch pandas sqlalchemy tqdm
```

### 2. 基础使用 (Basic Usage)

```python
from engine import AShareAlphaEngine

# 创建引擎（使用模拟数据）
engine = AShareAlphaEngine(use_simulated_data=True)

# 训练模型发现因子
engine.train(steps=500)

# 评估最佳策略
engine.evaluate_best_strategy()
```

### 3. 运行示例 (Run Examples)

```bash
# 示例 1: 快速测试
python example_usage.py --example 1

# 示例 4: 手动测试因子
python example_usage.py --example 4

# 示例 5: 批量测试因子库
python example_usage.py --example 5
```

### 4. 连接真实数据库 (Connect to Real Database)

```bash
# 设置环境变量
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_NAME=a_share_tick

# 使用真实数据
python -c "
from engine import AShareAlphaEngine
engine = AShareAlphaEngine(use_simulated_data=False)
engine.train()
"
```

---

## 📊 数据格式 (Data Format)

### Tick 数据表结构

```sql
CREATE TABLE tick_data (
    timestamp TIMESTAMP,          -- 时间戳
    stock_code VARCHAR(10),       -- 股票代码（如 SH600000）
    price DECIMAL(10, 2),         -- 最新价
    volume BIGINT,                -- 成交量
    buy_volume BIGINT,            -- 主动买入量
    sell_volume BIGINT,           -- 主动卖出量
    bid_price_1 DECIMAL(10, 2),   -- 买一价
    ask_price_1 DECIMAL(10, 2),   -- 卖一价
    bid_volume_1 BIGINT,          -- 买一量
    ask_volume_1 BIGINT           -- 卖一量
);
```

---

## 🔧 技术架构 (Technical Architecture)

### 1. 特征工程 (10 Features)

| 编号 | 特征名 | 说明 |
|-----|-------|------|
| F0 | Tick Return | 逐笔收益率 |
| F1 | Order Flow Imbalance | 订单流不平衡 |
| F2 | Bid-Ask Spread | 买卖价差 |
| F3 | VWAP Deviation | VWAP 偏离度 |
| F4 | Price Momentum | 价格动量 |
| F5 | Volume Surge | 成交量激增 |
| F6 | Tick Direction | Tick 方向强度 |
| F7 | Large Order Ratio | 大单占比 |
| F8 | Log Volume | 对数成交量 |
| F9 | Volume Ratio | 成交量比率 |

### 2. 操作算子 (27 Operators)

**基础运算 (9):** ADD, SUB, MUL, DIV, NEG, ABS, SIGN, SQUARE, SQRT  
**时序操作 (4):** DELAY1, DELAY5, DELTA1, DELTA5  
**滚动统计 (5):** MA5, MA10, MA20, STD10, STD20  
**滚动极值 (4):** MAX5, MAX10, MIN5, MIN10  
**统计分析 (2):** RANK10, RANK20  
**高级操作 (3):** GATE, DECAY, CORR  

### 3. 因子公式示例

使用栈式后缀表达式（Stack-based Postfix Notation）：

```python
# 动量因子: F0 - DELAY5(F0)
formula = [0, 0, 20, 11]  # F0, F0, DELAY5, SUB

# 订单流因子: F1 × F5
formula = [1, 5, 12]  # F1, F5, MUL

# 价差波动: STD10(F2)
formula = [2, 26]  # F2, STD10
```

### 4. A 股市场特性

- **T+1 交易**: 当天买入次日才能卖出
- **涨跌停限制**: ±10%（ST 股票 ±5%）
- **交易成本**:
  - 佣金: 0.03%（最低 5 元）
  - 印花税: 0.1%（仅卖出）

---

## 📈 评估指标 (Evaluation Metrics)

- **Total Return**: 累计收益率
- **Sharpe Ratio**: 夏普比率（风险调整收益）
- **Win Rate**: 胜率
- **Max Drawdown**: 最大回撤
- **Avg Trades**: 平均交易次数

---

## 💡 使用建议 (Best Practices)

1. **数据质量**: 建议使用 Level-2 高频行情数据
2. **样本外测试**: 使用滚动窗口避免过拟合
3. **风险控制**: 关注最大回撤和胜率指标
4. **实盘差异**: 考虑滑点、冲击成本等因素
5. **合规性**: 确保数据来源合法合规

---

## 📚 文档 (Documentation)

- **README.md** (本文档) - 快速开始指南
- **DETAILED_DOC.md** - 详细使用文档和配置说明
- **IMPLEMENTATION.md** - 实现总结和技术细节

---

## ⚠️ 免责声明 (Disclaimer)

本项目仅供学习和研究使用，不构成任何投资建议。使用本项目进行实盘交易的风险由使用者自行承担。

This project is for educational and research purposes only. It does not constitute investment advice. Users are responsible for any risks associated with real trading.

---

## 📧 联系方式 (Contact)

如有问题或建议，欢迎提交 Issue。

For questions or suggestions, please submit an Issue.

---

**Last Updated**: 2026-01-12  
**Version**: 1.0.0
