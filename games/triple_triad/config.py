"""MuZero configuration for Triple Triad."""
from typing import Optional
import pathlib


class MuZeroConfig:
    """Configuration class for MuZero training on Triple Triad.
    
    This class contains all the hyperparameters and configuration settings
    needed for training a MuZero agent on Triple Triad.
    
    Based on the tictactoe example from MuZero-general, adapted for
    the 3x3 board with card mechanics.
    """
    
    def __init__(self):
        # fmt: off
        
        ### Game Configuration
        self.seed = 0  # Seed for numpy and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use
        
        # Game dimensions
        self.observation_shape = (3, 3, 11)  # (height, width, channels)
        self.action_space = list(range(45))  # 5 cards * 9 positions = 45 actions
        self.players = list(range(2))  # Players 0 and 1
        self.stacked_observations = 0  # No previous observations needed for this simple game
        
        # Game rules
        self.max_moves = 9  # One card per position on 3x3 board
        self.num_players = 2
        
        # Evaluation
        self.muzero_player = 0  # MuZero plays first
        self.opponent = "expert"  # Hard coded agent for evaluation
        
        ### Self-Play Configuration (GPU-Optimized for RTX 3060Ti)
        self.num_workers = 8  # Moderate worker count for balance
        self.selfplay_on_gpu = True  # Critical: Use GPU for MCTS simulations (3-5x faster)
        self.max_memory_gb = 7  # Leave 1GB headroom on 8GB card
        self.num_simulations = 30  # Moderate increase from 15 to improve quality without overwhelming GPU
        self.discount = 1.0  # No discount since game always has 9 moves
        self.temperature_threshold = None  # Use softmax temperature function
        self.mixed_precision = True
        
        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        ### Network Configuration
        self.network = "resnet"  # Use residual network
        self.support_size = 10  # Value and reward encoding range
        
        # Residual Network architecture (optimized for Triple Triad's simple 3x3 board)
        self.downsample = False
        self.blocks = 1  # Reduced from 2 - sufficient for simple game
        self.channels = 24  # Reduced from 32 - sufficient for this problem
        self.reduced_channels_reward = 8  # Reduced from 16
        self.reduced_channels_value = 8  # Reduced from 16
        self.reduced_channels_policy = 8  # Reduced from 16
        self.resnet_fc_reward_layers = [12]  # Reduced from [16]
        self.resnet_fc_value_layers = [12]  # Reduced from [16]
        self.resnet_fc_policy_layers = [12]  # Reduced from [16]
        
        ### Training Configuration
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem
        self.save_model = True
        self.training_steps = 200000  # Reduced for faster iteration
        self.batch_size = 512  # Increased from 768, fits in 8GB VRAM with optimized network
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = True  # Assuming GPU available
        
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Learning rate schedule (adjusted for larger batch size)
        self.lr_init = 0.1  # Slightly lower for batch_size=128 stability
        self.lr_decay_rate = 0.794
        self.lr_decay_steps = 10000  # Decay to 0.001 by end of 200k run
        
        ### Replay Buffer Configuration (size-optimized for 9-move games)
        self.replay_buffer_size = 20000  # Increased for more diversity
        self.num_unroll_steps = 5  # Match max_moves exactly
        self.td_steps = 9  # Match max_moves exactly
        self.PER = True  # Prioritized Experience Replay
        self.PER_alpha = 0.5
        
        # Reanalyze - disabled for faster training iteration
        self.use_last_model_value = False  # Disable reanalyse to isolate training speed
        self.reanalyse_on_gpu = True
        self.reanalyse_workers = 2
        self.reanalyse_batch_size = 128

        # Replay Buffer Prefetching (performance optimization)
        # Set > 1 to prefetch batches for overlap between computation and data fetching
        self.replay_buffer_prefetch_count = 5  # Increased from 3 for more overlap

        # Self-Play Weight Caching Optimization (performance optimization)
        # Set > 1 to cache weights and update every N training steps instead of every game
        # Reduces Ray call overhead significantly (5x fewer calls per game)
        # Recommended: 50-100 for optimal performance
        # Set to None to disable (default behavior - update weights every game)
        self.self_play_weight_update_interval = 100  # Moderate value to balance Ray overhead vs freshness

        self.self_play_game_batch_size = 30  # Moderate batch size for game saves

        # Self-play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        
        # ============== PROFILING CONFIGURATION ==============
        # Self-play profiling settings for performance tuning
        self.enable_self_play_profiling = False  # Set to True to enable profiling
        self.profiling_log_interval = 10  # Log every N games
        self.profiling_output_dir = pathlib.Path(__file__).resolve().parents[1] / "results" / "profiling"
        self.profiling_gpu_samples = 5  # Number of GPU utilization samples per game
        self.profiling_track_ray_overhead = True  # Track Ray actor call overhead
        self.profiling_include_mcts_breakdown = True  # Track MCTS phase breakdown

        # fmt: on
    
    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """Return the temperature for visit softmax.
        
        Args:
            trained_steps: Number of training steps completed.
            
        Returns:
            Temperature value for action selection.
        """
        # Return 1 for the first few steps, then gradually reduce
        if trained_steps < 100:
            return 1.0
        elif trained_steps < 500:
            return 0.5
        else:
            return 0.25
    
    def get_uniform_network_config(self) -> dict:
        """Get network configuration for uniform network (initial training).
        
        Returns:
            Dictionary with network parameters.
        """
        return {
            "observation_shape": self.observation_shape,
            "action_space_size": len(self.action_space),
            "num_blocks": self.blocks,
            "num_channels": self.channels,
            "num_value_blocks": self.reduced_channels_value,
            "num_policy_blocks": self.reduced_channels_policy,
            "num_reward_blocks": self.reduced_channels_reward,
        }
