# A股因子挖掘项目 (A-Share Factor Mining Project)

基于 AlphaGPT 框架的中国 A 股市场逐笔成交数据因子挖掘系统。

A factor mining system for Chinese A-share market using tick-by-tick transaction data, based on the AlphaGPT framework.

---

## 🎯 项目简介 (Project Overview)

本项目参考 AlphaGPT 的符号回归思路，针对中国 A 股市场特点进行了全面适配，使用股票逐笔成交数据（Tick Data）进行高频因子挖掘。

This project adapts the AlphaGPT symbolic regression approach for the Chinese A-share market, using tick-by-tick transaction data for high-frequency factor discovery.

### 核心特点 (Key Features)

✅ **逐笔数据支持** - 处理高频 tick 级别的交易数据  
✅ **A股市场适配** - T+1 制度、涨跌停、交易成本（佣金+印花税）  
✅ **微观结构特征** - 订单流、买卖价差、VWAP 偏离等 10 种特征  
✅ **时序操作库** - 27 种专业操作算子（延迟、滚动统计、排名等）  
✅ **智能因子挖掘** - 基于强化学习的自动因子生成  
✅ **完整回测引擎** - 考虑真实交易成本的性能评估  

---

## 📁 项目结构 (Project Structure)

```
a_share_alpha/
├── README.md              # 详细文档
├── __init__.py           # 模块初始化
├── config.py             # 配置参数
├── tick_features.py      # Tick 特征工程
├── data_loader.py        # 数据加载器
├── ops.py               # 时序操作算子
├── vm.py                # 因子执行引擎
├── backtest.py          # 回测引擎
├── alphagpt.py          # AlphaGPT 模型
├── engine.py            # 主训练引擎
└── example_usage.py     # 使用示例
```

---

## 🚀 快速开始 (Quick Start)

### 安装依赖 (Install Dependencies)

```bash
pip install torch pandas sqlalchemy tqdm
```

### 运行示例 (Run Examples)

```bash
# 示例 1: 快速测试（使用模拟数据）
python a_share_alpha/example_usage.py --example 1

# 示例 2: 自定义配置
python a_share_alpha/example_usage.py --example 2

# 示例 3: 连接数据库
python a_share_alpha/example_usage.py --example 3

# 示例 4: 手动测试因子
python a_share_alpha/example_usage.py --example 4

# 示例 5: 批量测试因子库
python a_share_alpha/example_usage.py --example 5
```

### Python 代码示例 (Python Code Example)

```python
from a_share_alpha.engine import AShareAlphaEngine

# 创建引擎（使用模拟数据）
engine = AShareAlphaEngine(use_simulated_data=True)

# 训练模型发现因子
engine.train(steps=500)

# 评估最佳策略
engine.evaluate_best_strategy()
```

---

## 📊 数据格式 (Data Format)

### Tick 数据表结构

```sql
CREATE TABLE tick_data (
    timestamp TIMESTAMP,       -- 时间戳
    stock_code VARCHAR(10),   -- 股票代码 (如 SH600000)
    price DECIMAL(10, 2),     -- 最新价
    volume BIGINT,            -- 成交量
    buy_volume BIGINT,        -- 主动买入量
    sell_volume BIGINT,       -- 主动卖出量
    bid_price_1 DECIMAL(10, 2),   -- 买一价
    ask_price_1 DECIMAL(10, 2),   -- 卖一价
    bid_volume_1 BIGINT,      -- 买一量
    ask_volume_1 BIGINT       -- 卖一量
);
```

### 环境变量配置

```bash
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_NAME=a_share_tick
```

---

## 🔧 技术架构 (Technical Architecture)

### 1. 特征工程 (Feature Engineering)

从 Tick 数据提取 10 个特征：

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

### 2. 操作算子 (Operators)

27 种时序操作算子：

- **基础运算**: ADD, SUB, MUL, DIV, NEG, ABS, SIGN, SQUARE, SQRT
- **时序操作**: DELAY1, DELAY5, DELTA1, DELTA5
- **滚动统计**: MA5, MA10, MA20, STD10, STD20
- **滚动极值**: MAX5, MAX10, MIN5, MIN10
- **统计分析**: RANK10, RANK20
- **高级操作**: GATE, DECAY, CORR

### 3. 因子公式示例

```
动量因子: F0 DELAY5 SUB          # 当前收益 - 5期前收益
订单流因子: F1 F5 MUL             # 订单流 × 成交量激增
价差因子: F2 MA10 RANK20         # 价差移动平均的排名
```

### 4. A股市场特性

- **T+1 交易**: 当天买入次日才能卖出
- **涨跌停限制**: ±10%（ST 股票 ±5%）
- **交易成本**:
  - 佣金: 0.03% (最低 5 元)
  - 印花税: 0.1% (仅卖出)

---

## 📈 评估指标 (Evaluation Metrics)

- **Total Return**: 累计收益率
- **Sharpe Ratio**: 夏普比率（风险调整收益）
- **Win Rate**: 胜率
- **Max Drawdown**: 最大回撤
- **Avg Trades**: 平均交易次数

---

## 🎓 与原版 AlphaGPT 的对比

| 维度 | 原版 (Crypto) | A股版本 |
|-----|--------------|---------|
| 市场 | 加密货币 | A股市场 |
| 数据频率 | 日线/分钟线 | Tick 逐笔 |
| 特征数量 | 6 | 10 |
| 操作算子 | 12 | 27 |
| 交易制度 | 24/7, T+0 | T+1, 涨跌停 |
| 交易成本 | 0.5% | 佣金+印花税 |
| 市场特性 | 高波动 | 监管市场 |

---

## 📝 使用建议 (Best Practices)

1. **数据质量**: 建议使用 Level-2 高频行情数据
2. **样本外测试**: 使用滚动窗口避免过拟合
3. **风险控制**: 关注最大回撤和胜率
4. **实盘差异**: 考虑滑点、冲击成本等因素
5. **合规性**: 确保数据来源合法合规

---

## ⚠️ 免责声明 (Disclaimer)

本项目仅供学习和研究使用，不构成任何投资建议。使用本项目进行实盘交易的风险由使用者自行承担。

This project is for educational and research purposes only. It does not constitute investment advice. Users are responsible for any risks associated with real trading.

---

## 📚 参考资料 (References)

- [AlphaGPT 原始项目](https://github.com/L1CH333/AlphaGPT)
- WorldQuant 101 Alphas
- A-Share Market Microstructure Research

---

## 📧 联系方式 (Contact)

如有问题或建议，欢迎提交 Issue 或 Pull Request。

For questions or suggestions, please submit an Issue or Pull Request.

---

**Last Updated**: 2026-01-12
