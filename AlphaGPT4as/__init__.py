"""
AlphaGPT4as - A-Share Factor Mining Module

基于深度强化学习的 A 股因子挖掘系统
A deep reinforcement learning-based factor mining system for Chinese A-share market
using tick-by-tick transaction data.
"""

__version__ = "1.0.0"

from .config import AShareConfig
from .engine import AShareAlphaEngine
from .data_loader import AShareTickDataLoader
from .alphagpt import AShareAlphaGPT
from .backtest import AShareBacktest
from .vm import AShareStackVM

__all__ = [
    'AShareConfig',
    'AShareAlphaEngine',
    'AShareTickDataLoader',
    'AShareAlphaGPT',
    'AShareBacktest',
    'AShareStackVM',
]
