import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """Time series delay operator"""
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """Time series delta (difference)"""
    return x - _ts_delay(x, d)

@torch.jit.script
def _ts_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling sum"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).sum(dim=-1)

@torch.jit.script
def _ts_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling mean"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).mean(dim=-1)

@torch.jit.script
def _ts_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling standard deviation"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).std(dim=-1)

@torch.jit.script
def _ts_rank(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling rank (percentile position in window)"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, window, 1)
    
    # Calculate rank for each window
    current_val = x.unsqueeze(-1)
    
    # Find position in sorted window
    rank = (windows < current_val).float().sum(dim=-1)
    rank = rank / (window - 1)  # Normalize to [0, 1]
    
    return rank

@torch.jit.script
def _ts_max(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling maximum"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).max(dim=-1)[0]

@torch.jit.script
def _ts_min(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling minimum"""
    pad = torch.zeros((x.shape[0], window-1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).min(dim=-1)[0]

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Conditional gate: if condition > 0 then x else y"""
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    """Exponential decay combination"""
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2) + 0.4 * _ts_delay(x, 3)

@torch.jit.script
def _op_correlation(x: torch.Tensor, y: torch.Tensor, window: int = 20) -> torch.Tensor:
    """Rolling correlation between x and y"""
    # Simplified correlation using covariance
    x_mean = _ts_mean(x, window)
    y_mean = _ts_mean(y, window)
    
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov = _ts_mean(x_centered * y_centered, window)
    x_std = _ts_std(x, window) + 1e-6
    y_std = _ts_std(y, window) + 1e-6
    
    corr = cov / (x_std * y_std)
    return torch.clamp(corr, -1.0, 1.0)

# A-Share specific operators configuration
# Format: (name, function, arity)
OPS_CONFIG = [
    # Basic arithmetic
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    
    # Unary operations
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('SQUARE', lambda x: x * x, 1),
    ('SQRT', lambda x: torch.sqrt(torch.abs(x) + 1e-6), 1),
    
    # Time series operators
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('DELAY5', lambda x: _ts_delay(x, 5), 1),
    ('DELTA1', lambda x: _ts_delta(x, 1), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    
    # Rolling aggregations
    ('MA5', lambda x: _ts_mean(x, 5), 1),
    ('MA10', lambda x: _ts_mean(x, 10), 1),
    ('MA20', lambda x: _ts_mean(x, 20), 1),
    ('STD10', lambda x: _ts_std(x, 10), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    
    # Rolling extrema
    ('MAX5', lambda x: _ts_max(x, 5), 1),
    ('MAX10', lambda x: _ts_max(x, 10), 1),
    ('MIN5', lambda x: _ts_min(x, 5), 1),
    ('MIN10', lambda x: _ts_min(x, 10), 1),
    
    # Statistical operators
    ('RANK10', lambda x: _ts_rank(x, 10), 1),
    ('RANK20', lambda x: _ts_rank(x, 20), 1),
    
    # Advanced operators
    ('GATE', _op_gate, 3),
    ('DECAY', _op_decay, 1),
    ('CORR', lambda x, y: _op_correlation(x, y, 20), 2),
]
