import torch
import numpy as np

class TickIndicators:
    """A-Share tick-level market microstructure indicators"""
    
    @staticmethod
    def order_flow_imbalance(buy_volume, sell_volume):
        """
        订单流不平衡度
        Args:
            buy_volume: 主动买入成交量
            sell_volume: 主动卖出成交量
        Returns:
            order flow imbalance in [-1, 1]
        """
        total_volume = buy_volume + sell_volume + 1e-9
        ofi = (buy_volume - sell_volume) / total_volume
        return torch.clamp(ofi, -1.0, 1.0)
    
    @staticmethod
    def bid_ask_spread(bid_price, ask_price, mid_price):
        """
        买卖价差 (相对于中间价)
        Args:
            bid_price: 买一价
            ask_price: 卖一价
            mid_price: 中间价
        Returns:
            relative spread
        """
        spread = (ask_price - bid_price) / (mid_price + 1e-9)
        return torch.clamp(spread, 0.0, 0.1)  # Cap at 10%
    
    @staticmethod
    def volume_weighted_price_deviation(price, volume, window=20):
        """
        成交量加权价格偏离度
        Args:
            price: tick价格
            volume: tick成交量
            window: 滚动窗口
        Returns:
            deviation from VWAP
        """
        # Pad for rolling window
        pad = torch.zeros((price.shape[0], window-1), device=price.device)
        p_pad = torch.cat([pad, price], dim=1)
        v_pad = torch.cat([pad, volume], dim=1)
        
        # Calculate VWAP
        p_windows = p_pad.unfold(1, window, 1)
        v_windows = v_pad.unfold(1, window, 1)
        
        vwap = (p_windows * v_windows).sum(dim=-1) / (v_windows.sum(dim=-1) + 1e-9)
        
        deviation = (price - vwap) / (vwap + 1e-9)
        return torch.clamp(deviation, -0.2, 0.2)
    
    @staticmethod
    def price_momentum(price, window=10):
        """
        价格动量
        Args:
            price: tick价格
            window: 回看窗口
        Returns:
            momentum
        """
        price_lag = torch.roll(price, window, dims=1)
        momentum = (price - price_lag) / (price_lag + 1e-9)
        momentum[:, :window] = 0  # Zero out invalid positions
        return torch.clamp(momentum, -0.2, 0.2)
    
    @staticmethod
    def volume_surge(volume, window=30):
        """
        成交量激增指标
        Args:
            volume: tick成交量
            window: 基准窗口
        Returns:
            volume surge ratio
        """
        pad = torch.zeros((volume.shape[0], window-1), device=volume.device)
        v_pad = torch.cat([pad, volume], dim=1)
        
        avg_vol = v_pad.unfold(1, window, 1).mean(dim=-1)
        surge = volume / (avg_vol + 1e-9)
        
        return torch.clamp(torch.log1p(surge), 0.0, 5.0)
    
    @staticmethod
    def tick_direction_strength(price):
        """
        tick方向强度 (上涨tick vs 下跌tick)
        Args:
            price: tick价格序列
        Returns:
            directional strength
        """
        price_diff = price - torch.roll(price, 1, dims=1)
        price_diff[:, 0] = 0
        
        up_ticks = (price_diff > 0).float()
        down_ticks = (price_diff < 0).float()
        
        # Rolling sum over window
        window = 20
        pad = torch.zeros((price.shape[0], window-1), device=price.device)
        up_pad = torch.cat([pad, up_ticks], dim=1)
        down_pad = torch.cat([pad, down_ticks], dim=1)
        
        up_count = up_pad.unfold(1, window, 1).sum(dim=-1)
        down_count = down_pad.unfold(1, window, 1).sum(dim=-1)
        
        strength = (up_count - down_count) / (window + 1e-9)
        return strength
    
    @staticmethod
    def large_order_ratio(volume, large_threshold_percentile=90):
        """
        大单占比
        Args:
            volume: tick成交量
            large_threshold_percentile: 大单阈值百分位
        Returns:
            ratio of large orders
        Note:
            For production use with large datasets, consider using a rolling quantile
            or pre-computed threshold to improve performance.
        """
        # Calculate threshold (simplified version using global quantile)
        threshold = torch.quantile(volume, large_threshold_percentile / 100.0, dim=1, keepdim=True)
        large_orders = (volume > threshold).float()
        
        # Rolling average
        window = 20
        pad = torch.zeros((large_orders.shape[0], window-1), device=large_orders.device)
        lo_pad = torch.cat([pad, large_orders], dim=1)
        
        ratio = lo_pad.unfold(1, window, 1).mean(dim=-1)
        return ratio


class AShareFeatureEngineer:
    """Feature engineering for A-Share tick data"""
    
    INPUT_DIM = 10
    
    @staticmethod
    def compute_features(raw_tick_dict):
        """
        Compute features from raw tick data
        
        Args:
            raw_tick_dict: Dictionary containing:
                - 'price': last price
                - 'volume': tick volume
                - 'buy_volume': active buy volume
                - 'sell_volume': active sell volume
                - 'bid_price': best bid price
                - 'ask_price': best ask price
                - 'bid_volume': bid volume
                - 'ask_volume': ask volume
        
        Returns:
            feature tensor of shape [N_stocks, INPUT_DIM, N_ticks]
        """
        price = raw_tick_dict['price']
        volume = raw_tick_dict['volume']
        buy_vol = raw_tick_dict['buy_volume']
        sell_vol = raw_tick_dict['sell_volume']
        bid_price = raw_tick_dict['bid_price']
        ask_price = raw_tick_dict['ask_price']
        
        # Calculate mid price
        mid_price = (bid_price + ask_price) / 2.0
        
        # 1. Tick returns
        ret = torch.log(price / (torch.roll(price, 1, dims=1) + 1e-9))
        ret[:, 0] = 0
        
        # 2. Order flow imbalance
        ofi = TickIndicators.order_flow_imbalance(buy_vol, sell_vol)
        
        # 3. Bid-ask spread
        spread = TickIndicators.bid_ask_spread(bid_price, ask_price, mid_price)
        
        # 4. VWAP deviation
        vwap_dev = TickIndicators.volume_weighted_price_deviation(price, volume)
        
        # 5. Price momentum
        momentum = TickIndicators.price_momentum(price, window=10)
        
        # 6. Volume surge
        vol_surge = TickIndicators.volume_surge(volume)
        
        # 7. Tick direction strength
        tick_strength = TickIndicators.tick_direction_strength(price)
        
        # 8. Large order ratio
        large_ratio = TickIndicators.large_order_ratio(volume)
        
        # 9. Normalized log volume
        log_vol = torch.log1p(volume)
        
        # 10. Volume ratio (current vs avg)
        avg_vol = torch.roll(volume, 1, dims=1)
        avg_vol[:, 0] = volume[:, 0]
        vol_ratio = volume / (avg_vol + 1e-9)
        
        # Robust normalization function
        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)
        
        # Stack all features
        features = torch.stack([
            robust_norm(ret),           # 0: tick return
            robust_norm(ofi),           # 1: order flow imbalance
            robust_norm(spread),        # 2: bid-ask spread
            robust_norm(vwap_dev),      # 3: VWAP deviation
            robust_norm(momentum),      # 4: price momentum
            robust_norm(vol_surge),     # 5: volume surge
            robust_norm(tick_strength), # 6: tick direction strength
            robust_norm(large_ratio),   # 7: large order ratio
            robust_norm(log_vol),       # 8: log volume
            robust_norm(vol_ratio)      # 9: volume ratio
        ], dim=1)
        
        return features
