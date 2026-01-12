import torch

try:
    from .ops import OPS_CONFIG
    from .tick_features import AShareFeatureEngineer
except ImportError:
    from ops import OPS_CONFIG
    from tick_features import AShareFeatureEngineer

class AShareStackVM:
    """Stack-based virtual machine for executing factor formulas on A-Share data"""
    
    def __init__(self):
        self.feat_offset = AShareFeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
    
    def execute(self, formula_tokens, feat_tensor):
        """
        Execute a factor formula on feature tensor
        
        Args:
            formula_tokens: List of token indices
            feat_tensor: Feature tensor of shape [N_stocks, N_features, N_ticks]
        
        Returns:
            Result tensor of shape [N_stocks, N_ticks] or None if invalid
        """
        stack = []
        
        try:
            for token in formula_tokens:
                token = int(token)
                
                # Token is a feature index
                if token < self.feat_offset:
                    # Extract feature: shape [N_stocks, N_ticks]
                    stack.append(feat_tensor[:, token, :])
                
                # Token is an operator
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    
                    # Check if enough arguments on stack
                    if len(stack) < arity:
                        return None
                    
                    # Pop arguments
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()  # Restore original order
                    
                    # Execute operator
                    func = self.op_map[token]
                    res = func(*args)
                    
                    # Handle NaN/Inf
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    stack.append(res)
                
                else:
                    # Invalid token
                    return None
            
            # Formula should produce exactly one result
            if len(stack) == 1:
                return stack[0]
            else:
                return None
                
        except Exception as e:
            # Any execution error returns None
            return None
