import pandas as pd
import torch
import sqlalchemy
from .config import AShareConfig
from .tick_features import AShareFeatureEngineer

class AShareTickDataLoader:
    """Data loader for A-Share tick-by-tick transaction data"""
    
    def __init__(self):
        self.engine = None  # Lazy initialization
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.stock_codes = None
        
    def load_data(self, limit_stocks=100, date_range=None):
        """
        Load tick data from database
        
        Args:
            limit_stocks: Number of stocks to load
            date_range: Tuple of (start_date, end_date) or None for recent data
        """
        print("Loading tick data from SQL...")
        
        # Initialize database engine if not already done
        if self.engine is None:
            try:
                self.engine = sqlalchemy.create_engine(AShareConfig.DB_URL)
            except Exception as e:
                print(f"Failed to create database engine: {e}")
                print("Using simulated data instead...")
                return self._load_simulated_data(limit_stocks)
        
        # Get top liquid stocks
        stock_query = f"""
        SELECT DISTINCT stock_code 
        FROM tick_data 
        WHERE volume > {AShareConfig.MIN_VOLUME_THRESHOLD}
        ORDER BY volume DESC
        LIMIT {limit_stocks}
        """
        
        try:
            stock_df = pd.read_sql(stock_query, self.engine)
            self.stock_codes = stock_df['stock_code'].tolist()
        except Exception as e:
            print(f"Database query failed: {e}")
            print("Using simulated data for demonstration...")
            return self._load_simulated_data(limit_stocks)
        
        if not self.stock_codes:
            raise ValueError("No stocks found in database.")
        
        # Build WHERE clause for stocks
        stock_str = "'" + "','".join(self.stock_codes) + "'"
        
        # Build date filter
        date_filter = ""
        if date_range:
            date_filter = f"AND date BETWEEN '{date_range[0]}' AND '{date_range[1]}'"
        
        # Query tick data
        tick_query = f"""
        SELECT 
            timestamp,
            stock_code,
            price,
            volume,
            buy_volume,
            sell_volume,
            bid_price_1 as bid_price,
            ask_price_1 as ask_price,
            bid_volume_1 as bid_volume,
            ask_volume_1 as ask_volume
        FROM tick_data
        WHERE stock_code IN ({stock_str})
        {date_filter}
        ORDER BY timestamp ASC
        """
        
        print("Querying tick data...")
        df = pd.read_sql(tick_query, self.engine)
        
        if df.empty:
            print("No tick data found, using simulated data...")
            return self._load_simulated_data(limit_stocks)
        
        print(f"Loaded {len(df)} tick records")
        
        # Convert to tensor format
        self.raw_data_cache = self._process_tick_dataframe(df)
        
        # Compute features
        self.feat_tensor = AShareFeatureEngineer.compute_features(self.raw_data_cache)
        
        # Compute forward returns as target
        self._compute_target_returns()
        
        print(f"Data Ready. Feature shape: {self.feat_tensor.shape}")
        
    def _process_tick_dataframe(self, df):
        """Convert pandas DataFrame to tensor format"""
        
        def to_tensor(col):
            pivot = df.pivot(index='timestamp', columns='stock_code', values=col)
            pivot = pivot.ffill().bfill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=AShareConfig.DEVICE)
        
        return {
            'price': to_tensor('price'),
            'volume': to_tensor('volume'),
            'buy_volume': to_tensor('buy_volume'),
            'sell_volume': to_tensor('sell_volume'),
            'bid_price': to_tensor('bid_price'),
            'ask_price': to_tensor('ask_price'),
            'bid_volume': to_tensor('bid_volume'),
            'ask_volume': to_tensor('ask_volume')
        }
    
    def _compute_target_returns(self):
        """Compute forward returns as prediction target"""
        # Use 10-period forward return as default
        price = self.raw_data_cache['price']
        
        # Forward 10-tick return
        t_current = price
        t_forward = torch.roll(price, -10, dims=1)
        
        self.target_ret = torch.log(t_forward / (t_current + 1e-9))
        
        # Zero out last 10 positions (no valid forward return)
        self.target_ret[:, -10:] = 0.0
    
    def _load_simulated_data(self, n_stocks=100, n_ticks=240):
        """
        Generate simulated tick data for testing/demonstration
        
        Args:
            n_stocks: Number of stocks to simulate
            n_ticks: Number of tick timestamps
        """
        print(f"Generating simulated tick data: {n_stocks} stocks, {n_ticks} ticks")
        
        torch.manual_seed(42)
        device = AShareConfig.DEVICE
        
        # Simulate price with random walk
        base_price = torch.rand(n_stocks, 1, device=device) * 50 + 10  # 10-60 RMB
        returns = torch.randn(n_stocks, n_ticks, device=device) * 0.002  # 0.2% volatility per tick
        price = base_price * torch.exp(torch.cumsum(returns, dim=1))
        
        # Simulate volume with positive values
        volume = torch.abs(torch.randn(n_stocks, n_ticks, device=device)) * 10000 + 5000
        
        # Simulate buy/sell split
        buy_ratio = torch.sigmoid(torch.randn(n_stocks, n_ticks, device=device))
        buy_volume = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)
        
        # Simulate bid-ask spread
        spread_bps = torch.rand(n_stocks, n_ticks, device=device) * 20 + 5  # 5-25 bps
        bid_price = price * (1 - spread_bps / 20000)
        ask_price = price * (1 + spread_bps / 20000)
        
        # Simulate bid/ask volumes
        bid_volume = torch.abs(torch.randn(n_stocks, n_ticks, device=device)) * 5000 + 2000
        ask_volume = torch.abs(torch.randn(n_stocks, n_ticks, device=device)) * 5000 + 2000
        
        self.stock_codes = [f"SH{600000+i:06d}" for i in range(n_stocks)]
        
        self.raw_data_cache = {
            'price': price,
            'volume': volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }
        
        # Compute features
        self.feat_tensor = AShareFeatureEngineer.compute_features(self.raw_data_cache)
        
        # Compute target returns
        self._compute_target_returns()
        
        print(f"Simulated Data Ready. Feature shape: {self.feat_tensor.shape}")
