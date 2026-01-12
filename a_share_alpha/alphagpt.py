import torch
import torch.nn as nn
from .config import AShareConfig
from .ops import OPS_CONFIG

class AShareAlphaGPT(nn.Module):
    """
    AlphaGPT model adapted for A-Share factor mining
    Uses transformer architecture to generate factor formulas
    """
    
    def __init__(self):
        super().__init__()
        
        # Model dimensions
        self.d_model = 128  # Increased for more complex patterns
        
        # Vocabulary: features + operators
        self.features_list = [f'F{i}' for i in range(AShareConfig.INPUT_DIM)]
        self.ops_list = [cfg[0] for cfg in OPS_CONFIG]
        
        self.vocab = self.features_list + self.ops_list
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Features: {self.features_list}")
        print(f"Operators: {self.ops_list}")
        
        # Embedding layers
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, AShareConfig.MAX_FORMULA_LEN + 1, self.d_model)
        )
        
        # Transformer decoder
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=3)
        
        # Output heads
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head_actor = nn.Linear(self.d_model, self.vocab_size)  # Policy network
        self.head_critic = nn.Linear(self.d_model, 1)  # Value network
        
    def forward(self, idx):
        """
        Forward pass
        
        Args:
            idx: Token indices of shape [Batch, SeqLen]
        
        Returns:
            logits: Action logits of shape [Batch, VocabSize]
            value: State value of shape [Batch, 1]
        """
        B, T = idx.size()
        
        # Embedding
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        
        # Transformer
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        
        # Get last position embedding for next token prediction
        last_emb = x[:, -1, :]
        
        # Policy and value heads
        logits = self.head_actor(last_emb)
        value = self.head_critic(last_emb)
        
        return logits, value
