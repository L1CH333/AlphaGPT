import torch
import os

class AShareConfig:
    """Configuration for A-Share factor mining"""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Database configuration for tick data
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','a_share_tick')}"
    
    # Training parameters
    BATCH_SIZE = 64  # Reduced for memory efficiency
    TRAIN_STEPS = 2000
    MAX_FORMULA_LEN = 10  # Reduced for faster training
    
    # A-Share market specific parameters
    TRADE_SIZE_CNY = 100000.0  # 10万元人民币标准手
    MIN_VOLUME_THRESHOLD = 1000000.0  # 最小成交量阈值(股)
    
    # Transaction costs (A股特有)
    COMMISSION_RATE = 0.0003  # 佣金 0.03%
    STAMP_TAX = 0.001  # 印花税 0.1% (仅卖出)
    MIN_COMMISSION = 5.0  # 最低佣金 5元
    
    # Trading rules
    TICK_RISE_LIMIT = 0.10  # 涨停 10%
    TICK_FALL_LIMIT = -0.10  # 跌停 10%
    ST_RISE_LIMIT = 0.05  # ST股涨停 5%
    ST_FALL_LIMIT = -0.05  # ST股跌停 5%
    
    # Feature dimensions (tick-level features)
    INPUT_DIM = 10  # Tick price, volume, bid-ask spread, order flow, etc.
    
    # Data parameters
    TICK_WINDOW = 240  # 4小时交易时间的tick数据窗口 (假设每分钟取1个tick)
    FORWARD_RETURN_PERIODS = [5, 10, 20]  # 预测未来5/10/20分钟收益率
