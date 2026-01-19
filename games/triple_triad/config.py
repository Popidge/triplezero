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
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use
        
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
        
        ### Self-Play Configuration
        self.num_workers = 1  # Number of simultaneous self-play workers
        self.selfplay_on_gpu = False
        self.num_simulations = 25  # Number of future moves to simulate
        self.discount = 1.0  # No discount since game always has 9 moves
        self.temperature_threshold = None  # Use softmax temperature function
        
        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        ### Network Configuration
        self.network = "resnet"  # Use residual network
        self.support_size = 10  # Value and reward encoding range
        
        # Residual Network architecture
        self.downsample = False
        self.blocks = 1  # Number of residual blocks
        self.channels = 16  # Number of channels
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.resnet_fc_reward_layers = [8]
        self.resnet_fc_value_layers = [8]
        self.resnet_fc_policy_layers = [8]
        
        ### Training Configuration
        self.results_path = pathlib.Path(__file__).resolve().parents[2] / "results" / pathlib.Path(__file__).stem
        self.save_model = True
        self.training_steps = 1000000
        self.batch_size = 64
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = True  # Assuming GPU available
        
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Learning rate schedule
        self.lr_init = 0.003
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000
        
        ### Replay Buffer Configuration
        self.replay_buffer_size = 3000
        self.num_unroll_steps = 20
        self.td_steps = 20
        self.PER = True  # Prioritized Experience Replay
        self.PER_alpha = 0.5
        
        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False
        
        # Self-play / training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None
        
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
