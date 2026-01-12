import torch

try:
    from .config import AShareConfig
except ImportError:
    from config import AShareConfig

class AShareBacktest:
    """
    Backtesting engine for A-Share market
    Implements A-Share specific trading rules and costs
    """
    
    def __init__(self):
        self.trade_size = AShareConfig.TRADE_SIZE_CNY
        self.min_volume = AShareConfig.MIN_VOLUME_THRESHOLD
        
        # Transaction costs
        self.commission_rate = AShareConfig.COMMISSION_RATE
        self.stamp_tax = AShareConfig.STAMP_TAX
        self.min_commission = AShareConfig.MIN_COMMISSION
        
    def evaluate(self, factors, raw_data, target_ret):
        """
        Evaluate a factor's performance
        
        Args:
            factors: Factor values of shape [N_stocks, N_ticks]
            raw_data: Dictionary of raw tick data
            target_ret: Target forward returns [N_stocks, N_ticks]
        
        Returns:
            (fitness_score, avg_return)
        """
        volume = raw_data['volume']
        price = raw_data['price']
        
        # Generate trading signal from factor
        # Normalize factor to [0, 1] using sigmoid
        signal = torch.sigmoid(factors)
        
        # Filter by volume (can't trade illiquid stocks)
        is_tradable = (volume > self.min_volume).float()
        
        # Position: long if signal > 0.7, considering tradability
        position = (signal > 0.7).float() * is_tradable
        
        # Calculate transaction costs
        # Get previous position for turnover calculation
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0  # No position at t=0
        
        # Turnover (position change)
        turnover = torch.abs(position - prev_pos)
        
        # Commission (both buy and sell)
        commission_cost = turnover * self.commission_rate
        
        # Apply minimum commission (5 RMB per trade)
        # Simplified: use ratio-based minimum
        min_commission_ratio = self.min_commission / self.trade_size
        commission_cost = torch.maximum(
            commission_cost, 
            turnover * min_commission_ratio
        )
        
        # Stamp tax (only on sell)
        sell_trades = torch.minimum(turnover, prev_pos)  # Can only sell if had position
        stamp_cost = sell_trades * self.stamp_tax
        
        # Total transaction cost
        total_cost = commission_cost + stamp_cost
        
        # Gross PnL from position
        gross_pnl = position * target_ret
        
        # Net PnL after costs
        net_pnl = gross_pnl - total_cost
        
        # Calculate cumulative return per stock
        cum_ret = net_pnl.sum(dim=1)
        
        # Penalize large drawdowns (single-tick loss > 5%)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        
        # Risk-adjusted score
        score = cum_ret - (big_drawdowns * 1.0)
        
        # Penalize inactive strategies (must trade at least 10 times)
        activity = position.sum(dim=1)
        score = torch.where(
            activity < 10, 
            torch.tensor(-10.0, device=score.device), 
            score
        )
        
        # Use median score as fitness (robust to outliers)
        final_fitness = torch.median(score)
        
        return final_fitness, cum_ret.mean().item()
    
    def detailed_backtest(self, factors, raw_data, target_ret):
        """
        Perform detailed backtest with more metrics
        
        Returns:
            Dictionary with detailed backtest results
        """
        volume = raw_data['volume']
        price = raw_data['price']
        
        signal = torch.sigmoid(factors)
        is_tradable = (volume > self.min_volume).float()
        position = (signal > 0.7).float() * is_tradable
        
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        
        turnover = torch.abs(position - prev_pos)
        commission_cost = torch.maximum(
            turnover * self.commission_rate,
            turnover * (self.min_commission / self.trade_size)
        )
        
        sell_trades = torch.minimum(turnover, prev_pos)
        stamp_cost = sell_trades * self.stamp_tax
        total_cost = commission_cost + stamp_cost
        
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - total_cost
        
        # Calculate metrics
        cum_ret_per_stock = net_pnl.sum(dim=1)
        
        # Sharpe ratio (simplified)
        ret_mean = net_pnl.mean(dim=1)
        ret_std = net_pnl.std(dim=1) + 1e-6
        sharpe = ret_mean / ret_std
        
        # Win rate
        win_trades = (net_pnl > 0).float().sum(dim=1)
        total_trades = position.sum(dim=1) + 1e-6
        win_rate = win_trades / total_trades
        
        # Maximum drawdown (simplified)
        cumulative = torch.cumsum(net_pnl, dim=1)
        running_max = torch.cumsum(torch.maximum(net_pnl, torch.zeros_like(net_pnl)), dim=1)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max(dim=1)[0]
        
        return {
            'total_return': cum_ret_per_stock.mean().item(),
            'sharpe': sharpe.mean().item(),
            'win_rate': win_rate.mean().item(),
            'max_drawdown': max_drawdown.mean().item(),
            'avg_trades': total_trades.mean().item(),
            'net_pnl_tensor': net_pnl
        }
