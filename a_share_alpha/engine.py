import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os

from .config import AShareConfig
from .data_loader import AShareTickDataLoader
from .alphagpt import AShareAlphaGPT
from .vm import AShareStackVM
from .backtest import AShareBacktest

class AShareAlphaEngine:
    """
    Main engine for A-Share factor mining
    Uses reinforcement learning to discover profitable factors
    """
    
    def __init__(self, use_simulated_data=True):
        """
        Initialize the engine
        
        Args:
            use_simulated_data: If True, use simulated data instead of database
        """
        print("Initializing A-Share Alpha Engine...")
        
        # Load data
        self.loader = AShareTickDataLoader()
        if use_simulated_data:
            self.loader._load_simulated_data(n_stocks=100, n_ticks=240)
        else:
            self.loader.load_data(limit_stocks=100)
        
        # Initialize model
        self.model = AShareAlphaGPT().to(AShareConfig.DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        
        # Initialize VM and backtest engine
        self.vm = AShareStackVM()
        self.bt = AShareBacktest()
        
        # Track best strategy
        self.best_score = -float('inf')
        self.best_formula = None
        self.best_formula_readable = None
        
        print("Engine initialized successfully!")
        
    def train(self, steps=None):
        """
        Train the model to discover profitable factors
        
        Args:
            steps: Number of training steps (uses config default if None)
        """
        if steps is None:
            steps = AShareConfig.TRAIN_STEPS
        
        print(f"🚀 Starting A-Share Factor Mining for {steps} steps...")
        pbar = tqdm(range(steps))
        
        for step in pbar:
            bs = AShareConfig.BATCH_SIZE
            
            # Start with BOS token (index 0, represented as zeros)
            inp = torch.zeros((bs, 1), dtype=torch.long, device=AShareConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            # Generate factor formula autoregressively
            for _ in range(AShareConfig.MAX_FORMULA_LEN):
                logits, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                
                # Append to input for next step
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            # Stack all tokens: [Batch, MAX_FORMULA_LEN]
            seqs = torch.stack(tokens_list, dim=1)
            
            # Evaluate each formula
            rewards = torch.zeros(bs, device=AShareConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                # Execute formula on data
                res = self.vm.execute(formula, self.loader.feat_tensor)
                
                if res is None:
                    # Invalid formula
                    rewards[i] = -5.0
                    continue
                
                # Check if factor has variance (non-constant)
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                # Backtest the factor
                score, ret_val = self.bt.evaluate(
                    res, 
                    self.loader.raw_data_cache, 
                    self.loader.target_ret
                )
                
                rewards[i] = score
                
                # Track best strategy
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    self.best_formula_readable = self._formula_to_readable(formula)
                    
                    tqdm.write(
                        f"[NEW BEST] Score: {score:.4f} | Return: {ret_val:.4%} | "
                        f"Formula: {self.best_formula_readable}"
                    )
            
            # Compute policy gradient loss
            # Normalize rewards (advantage estimation)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            # Policy gradient: maximize expected reward
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()
            
            # Optimization step
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            
            # Update progress bar
            pbar.set_postfix({
                'AvgReward': f'{rewards.mean().item():.2f}',
                'BestScore': f'{self.best_score:.2f}'
            })
        
        # Save best strategy
        self._save_best_strategy()
        
        print(f"\n✅ Training complete!")
        print(f"Best Score: {self.best_score:.4f}")
        print(f"Best Formula: {self.best_formula_readable}")
        
    def _formula_to_readable(self, formula):
        """Convert formula tokens to readable string"""
        vocab = self.model.vocab
        tokens = [vocab[min(t, len(vocab)-1)] for t in formula]
        return ' '.join(tokens)
    
    def _save_best_strategy(self):
        """Save best strategy to file"""
        output = {
            'score': self.best_score,
            'formula_tokens': self.best_formula,
            'formula_readable': self.best_formula_readable,
            'config': {
                'max_formula_len': AShareConfig.MAX_FORMULA_LEN,
                'input_dim': AShareConfig.INPUT_DIM
            }
        }
        
        os.makedirs('output', exist_ok=True)
        output_path = 'output/best_ashare_strategy.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Best strategy saved to {output_path}")
    
    def evaluate_best_strategy(self):
        """Perform detailed evaluation of best strategy"""
        if self.best_formula is None:
            print("No strategy found yet. Please train first.")
            return
        
        print("\n📊 Detailed Evaluation of Best Strategy")
        print("=" * 60)
        
        # Execute formula
        res = self.vm.execute(self.best_formula, self.loader.feat_tensor)
        
        if res is None:
            print("Best formula is invalid!")
            return
        
        # Detailed backtest
        metrics = self.bt.detailed_backtest(
            res,
            self.loader.raw_data_cache,
            self.loader.target_ret
        )
        
        print(f"Total Return:    {metrics['total_return']:.4%}")
        print(f"Sharpe Ratio:    {metrics['sharpe']:.4f}")
        print(f"Win Rate:        {metrics['win_rate']:.4%}")
        print(f"Max Drawdown:    {metrics['max_drawdown']:.4%}")
        print(f"Avg Trades:      {metrics['avg_trades']:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    # Run the engine
    engine = AShareAlphaEngine(use_simulated_data=True)
    engine.train(steps=500)  # Shorter run for testing
    engine.evaluate_best_strategy()
