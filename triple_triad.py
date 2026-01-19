#!/usr/bin/env python3
# Auto-generated on 2026-01-19 23:37:06
# Generated from modular Triple Triad implementation
# Drop this file into MuZero's games/ directory

import datetime
import pathlib

import numpy as np
import torch

__version__ = "1.0.0"


# ============== CARD DEFINITIONS ==============
# Card list for Triple Triad
# Format: (N, E, S, W) - Top, Right, Bottom, Left values

TRIPLE_TRIAD_CARDS = {
    # Level 1 Cards
    "Geezard": (1, 4, 5, 1),
    "Fungar": (5, 1, 1, 3),
    "Bite Bug": (1, 3, 3, 5),
    "Red Bat": (6, 1, 1, 2),
    "Blobra": (2, 3, 1, 5),
    "Gayla": (2, 1, 4, 4),
    "Gesper": (1, 5, 4, 1),
    "Fastitocalon-F": (3, 5, 2, 1),
    "Blood Soul": (2, 1, 6, 1),
    "Caterchipillar": (4, 2, 4, 3),
    "Cockatrice": (6, 1, 2, 6),
    
    # Level 2 Cards
    "Grat": (7, 1, 3, 1),
    "Buel": (6, 2, 2, 3),
    "Mesmerize": (5, 3, 3, 4),
    "Glacial Eye": (6, 1, 4, 3),
    "Belhelmel": (3, 4, 5, 3),
    "Thrustaevis": (5, 3, 2, 5),
    "Anacondaur": (5, 1, 3, 5),
    "Creeps": (5, 2, 5, 2),
    "Grendel": (4, 4, 5, 2),
    "Jelleye": (3, 2, 1, 7),
    "Grand Mantis": (5, 2, 5, 3),
    
    # Level 3 Cards
    "Forbidden": (6, 6, 3, 2),
    "Armadodo": (6, 3, 1, 6),
    "TriFace": (3, 5, 5, 5),
    "Fastitcalon": (7, 5, 1, 3),
    "Snow Lion": (7, 1, 5, 3),
    "Ochu": (5, 6, 3, 3),
    "SAM08G": (5, 6, 2, 4),
    "Death Claw": (4, 4, 7, 2),
    "Cactaur": (6, 2, 6, 3),
    "Tonberry": (3, 6, 4, 4),
    "Abyss Worm": (7, 2, 3, 5),
    
    # Level 4 Cards
    "Turtapod": (2, 3, 6, 7),
    "Vysage": (6, 5, 4, 5),
    "T-Rexaur": (4, 6, 2, 7),
    "Bomb": (2, 7, 6, 3),
    "Blitz": (1, 6, 4, 7),
    "Wendigo": (7, 3, 1, 6),
    "Torama": (7, 4, 4, 4),
    "Imp": (3, 7, 3, 6),
    "Blue Dragon": (6, 2, 7, 3),
    "Adamantiose": (4, 5, 5, 6),
    "Hexadragon": (7, 5, 4, 3),
    
    # Level 5 Cards
    "Iron Giant": (6, 5, 6, 5),
    "Behemoth": (3, 6, 5, 7),
    "Chimera": (7, 6, 5, 3),
    "PuPu": (3, 10, 2, 1),
    "Elastiod": (6, 2, 6, 7),
    "GIM47N": (5, 5, 7, 4),
    "Malboro": (7, 7, 4, 2),
    "Ruby Dragon": (7, 2, 7, 4),
    "Elnoyle": (5, 3, 7, 6),
    "Tonberry King": (4, 6, 7, 4),
    "Wedge, Biggs": (6, 6, 2, 7),
    
    # Level 6 Cards
    "Fujin, Raijin": (2, 8, 8, 4),
    "Elvoret": (7, 8, 3, 4),
    "X-ATM092": (4, 8, 4, 3),
    "Grenaldo": (7, 2, 8, 5),
    "Gerogero": (1, 8, 8, 3),
    "Iguion": (8, 2, 8, 2),
    "Abadon": (6, 8, 4, 5),
    "Trauma": (4, 8, 5, 6),
    "Oilboyle": (1, 8, 4, 8),
    "Shumi Tribe": (6, 5, 8, 4),
    "Krysta": (7, 5, 8, 1),
    
    # Level 7 Cards
    "Propagator": (8, 4, 4, 8),
    "Jumbo Cactaur": (8, 8, 4, 4),
    "Tri-Point": (8, 5, 2, 8),
    "Gargantua": (5, 6, 6, 8),
    "Mobile Type 8": (8, 6, 7, 3),
    "Sphinxara": (8, 3, 5, 8),
    "Tiamat": (8, 8, 5, 4),
    "BGH251F2": (5, 7, 8, 5),
    "Red Giant": (6, 8, 4, 7),
    "Catoblepas": (1, 8, 7, 7),
    "Ultima Weapon": (7, 7, 2, 8),
    
    # Level 8 Cards
    "Chubby Chocobo": (4, 4, 8, 9),
    "Angelo": (9, 6, 7, 3),
    "Gilgamesh": (3, 7, 9, 6),
    "MinMog": (9, 3, 9, 2),
    "Chicobo": (9, 4, 8, 4),
    "Quezacotl": (2, 9, 9, 4),
    "Shiva": (6, 7, 4, 9),
    "Ifrit": (9, 6, 2, 8),
    "Siren": (9, 6, 2, 8),
    "Sacred": (5, 1, 9, 9),
    "Minotaur": (9, 9, 2, 5),
    
    # Level 9 Cards
    "Carbuncle": (8, 4, 10, 4),
    "Diablos": (5, 10, 8, 3),
    "Leviathan": (7, 10, 1, 7),
    "Odin": (8, 10, 3, 5),
    "Pandemonia": (10, 1, 7, 7),
    "Cerberus": (7, 4, 6, 10),
    "Alexander": (9, 10, 4, 2),
    "Phoenix": (7, 2, 7, 10),
    "Bahumut": (10, 8, 2, 6),
    "Doomtrain": (3, 1, 10, 10),
    "Eden": (4, 4, 9, 10),
    
    # Level 10 Cards
    "Ward": (10, 7, 2, 8),
    "Kiros": (6, 7, 6, 10),
    "Laguna": (5, 10, 3, 9),
    "Selphie": (10, 8, 4, 6),
    "Quistis": (9, 6, 10, 2),
    "Irvine": (2, 6, 9, 10),
    "Zell": (8, 5, 10, 6),
    "Rinoa": (4, 10, 2, 10),
    "Edea": (10, 10, 3, 3),
    "Seifer": (6, 9, 10, 4),
    "Squall": (10, 10, 9, 6),
}

# Card levels for dealing algorithm
# Used to implement balanced hand distribution:
# - 1 card from levels 1-2
# - 1 card from levels 3-4
# - 1 card from levels 5-6
# - 1 card from levels 7-8
# - 1 card from levels 9-10 (must be unique between players)

CARD_LEVELS = {
    # Level 1
    "Geezard": 1,
    "Fungar": 1,
    "Bite Bug": 1,
    "Red Bat": 1,
    "Blobra": 1,
    "Gayla": 1,
    "Gesper": 1,
    "Fastitocalon-F": 1,
    "Blood Soul": 1,
    "Caterchipillar": 1,
    "Cockatrice": 1,
    
    # Level 2
    "Grat": 2,
    "Buel": 2,
    "Mesmerize": 2,
    "Glacial Eye": 2,
    "Belhelmel": 2,
    "Thrustaevis": 2,
    "Anacondaur": 2,
    "Creeps": 2,
    "Grendel": 2,
    "Jelleye": 2,
    "Grand Mantis": 2,
    
    # Level 3
    "Forbidden": 3,
    "Armadodo": 3,
    "TriFace": 3,
    "Fastitcalon": 3,
    "Snow Lion": 3,
    "Ochu": 3,
    "SAM08G": 3,
    "Death Claw": 3,
    "Cactaur": 3,
    "Tonberry": 3,
    "Abyss Worm": 3,
    
    # Level 4
    "Turtapod": 4,
    "Vysage": 4,
    "T-Rexaur": 4,
    "Bomb": 4,
    "Blitz": 4,
    "Wendigo": 4,
    "Torama": 4,
    "Imp": 4,
    "Blue Dragon": 4,
    "Adamantiose": 4,
    "Hexadragon": 4,
    
    # Level 5
    "Iron Giant": 5,
    "Behemoth": 5,
    "Chimera": 5,
    "PuPu": 5,
    "Elastiod": 5,
    "GIM47N": 5,
    "Malboro": 5,
    "Ruby Dragon": 5,
    "Elnoyle": 5,
    "Tonberry King": 5,
    "Wedge, Biggs": 5,
    
    # Level 6
    "Fujin, Raijin": 6,
    "Elvoret": 6,
    "X-ATM092": 6,
    "Grenaldo": 6,
    "Gerogero": 6,
    "Iguion": 6,
    "Abadon": 6,
    "Trauma": 6,
    "Oilboyle": 6,
    "Shumi Tribe": 6,
    "Krysta": 6,
    
    # Level 7
    "Propagator": 7,
    "Jumbo Cactaur": 7,
    "Tri-Point": 7,
    "Gargantua": 7,
    "Mobile Type 8": 7,
    "Sphinxara": 7,
    "Tiamat": 7,
    "BGH251F2": 7,
    "Red Giant": 7,
    "Catoblepas": 7,
    "Ultima Weapon": 7,
    
    # Level 8
    "Chubby Chocobo": 8,
    "Angelo": 8,
    "Gilgamesh": 8,
    "MinMog": 8,
    "Chicobo": 8,
    "Quezacotl": 8,
    "Shiva": 8,
    "Ifrit": 8,
    "Siren": 8,
    "Sacred": 8,
    "Minotaur": 8,
    
    # Level 9
    "Carbuncle": 9,
    "Diablos": 9,
    "Leviathan": 9,
    "Odin": 9,
    "Pandemonia": 9,
    "Cerberus": 9,
    "Alexander": 9,
    "Phoenix": 9,
    "Bahumut": 9,
    "Doomtrain": 9,
    "Eden": 9,
    
    # Level 10
    "Ward": 10,
    "Kiros": 10,
    "Laguna": 10,
    "Selphie": 10,
    "Quistis": 10,
    "Irvine": 10,
    "Zell": 10,
    "Rinoa": 10,
    "Edea": 10,
    "Seifer": 10,
    "Squall": 10,
}



# ============== ABSTRACT GAME BASE ==============
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional
import numpy as np


class AbstractGame(ABC):
    """Base class that all MuZero games must inherit from.
    
    This class defines the interface that the MuZero algorithm expects from
    any game implementation. Subclasses must implement all abstract methods.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the game with an optional seed for reproducibility.
        
        Args:
            seed: Random seed for game initialization.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply an action to the game and return the new state.
        
        Args:
            action: The action to apply (from action_space).
            
        Returns:
            Tuple of (observation, reward, done) where:
                - observation: numpy array representing the game state
                - reward: float reward for the current player
                - done: boolean indicating if the game is over
        """
        pass

    @abstractmethod
    def to_play(self) -> int:
        """Return the current player.
        
        Returns:
            The current player (index from players list).
        """
        pass

    @abstractmethod
    def legal_actions(self) -> List[int]:
        """Return the list of legal actions for the current state.
        
        Returns:
            List of integers representing legal actions.
        """
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the game to its initial state.
        
        Returns:
            Initial observation of the game.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """Display the current game state."""
        pass

    def human_to_action(self) -> int:
        """Convert human input to an action.
        
        This method can be overridden for games that need human input.
        By default, raises NotImplementedError.
        
        Returns:
            The action selected by the human player.
        """
        raise NotImplementedError("Human input not implemented for this game.")

    def expert_agent(self) -> int:
        """Return an action from an expert agent.
        
        This method can be overridden to provide a hard-coded expert
        agent for evaluation purposes. By default, raises NotImplementedError.
        
        Returns:
            The action selected by the expert agent.
        """
        raise NotImplementedError("Expert agent not implemented for this game.")

    def action_to_string(self, action: int) -> str:
        """Convert an action number to a human-readable string.
        
        Args:
            action: The action number.
            
        Returns:
            String representation of the action.
        """
        return str(action)


# ============== MUZERO CONFIG ==============
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


# ============== CORE GAME LOGIC ==============
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import random


class TripleTriad:
    """Core game logic for Triple Triad.
    
    This class implements the complete game logic for Triple Triad with the
    "Open" ruleset (perfect information). It manages the board state, hands,
    card placement, and flipping mechanics.
    
    Attributes:
        card_names: List of card names for indexing.
        card_values: Dictionary mapping card name to (N, E, S, W) tuple.
        board: 3x3 numpy array storing (card_name, owner) tuples or None.
        hands: List of two lists, each containing 5 card names.
        current_player: 0 or 1 indicating which player's turn it is.
        moves_made: Number of moves made in the current game.
        seed: Random seed for reproducibility.
    """
    
    # Board dimensions
    BOARD_SIZE = 3
    NUM_CARDS_PER_PLAYER = 5
    TOTAL_CARDS = 10
    
    # Direction mappings for neighbor offsets and face indices
    # Format: (row_offset, col_offset, our_face_index, their_face_index)
    DIRECTIONS = {
        'N': (-1, 0, 0, 2),  # Our North vs their South
        'E': (0, 1, 1, 3),   # Our East vs their West
        'S': (1, 0, 2, 0),   # Our South vs their North
        'W': (0, -1, 3, 1),  # Our West vs their East
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the Triple Triad game.
        
        Args:
            seed: Optional random seed for reproducible game initialization.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.card_names = list(TRIPLE_TRIAD_CARDS.keys())
        self.card_values = TRIPLE_TRIAD_CARDS
        
        # Pre-compute all possible action templates for fast legal action lookup
        # Format: (card_idx, position) for all possible combinations
        self._action_templates = [
            (card_idx, pos) 
            for card_idx in range(self.NUM_CARDS_PER_PLAYER) 
            for pos in range(self.BOARD_SIZE * self.BOARD_SIZE)
        ]
        
        # Game state
        self.board: np.ndarray = np.full((3, 3), None, dtype=object)
        self.hands: List[List[str]] = [[], []]
        self.current_player: int = 0
        self.moves_made: int = 0
        
        # Initialize the game
        self._initialize_game()
    
    def _initialize_game(self) -> None:
        """Deal cards to players using balanced level distribution.
        
        Each player gets:
        - 1 card from levels 1-2
        - 1 card from levels 3-4
        - 1 card from levels 5-6
        - 1 card from levels 7-8
        - 1 card from levels 9-10 (must be unique between players)
        """
        # Group cards by level ranges
        level_1_2 = [c for c in self.card_names if CARD_LEVELS[c] <= 2]
        level_3_4 = [c for c in self.card_names if 3 <= CARD_LEVELS[c] <= 4]
        level_5_6 = [c for c in self.card_names if 5 <= CARD_LEVELS[c] <= 6]
        level_7_8 = [c for c in self.card_names if 7 <= CARD_LEVELS[c] <= 8]
        level_9_10 = [c for c in self.card_names if CARD_LEVELS[c] >= 9]
        
        # Shuffle each level group
        random.shuffle(level_1_2)
        random.shuffle(level_3_4)
        random.shuffle(level_5_6)
        random.shuffle(level_7_8)
        random.shuffle(level_9_10)
        
        # Deal to player 0: 1 card from each level range
        hand_0 = [
            level_1_2[0],
            level_3_4[0],
            level_5_6[0],
            level_7_8[0],
            level_9_10[0],
        ]
        
        # Deal to player 1: 1 card from each level range (9-10 must be unique)
        # Use index 1 for 9-10 to ensure uniqueness
        hand_1 = [
            level_1_2[1] if len(level_1_2) > 1 else level_1_2[0],
            level_3_4[1] if len(level_3_4) > 1 else level_3_4[0],
            level_5_6[1] if len(level_5_6) > 1 else level_5_6[0],
            level_7_8[1] if len(level_7_8) > 1 else level_7_8[0],
            level_9_10[1] if len(level_9_10) > 1 else level_9_10[0],
        ]
        
        # Shuffle each player's hand so cards aren't ordered by level
        random.shuffle(hand_0)
        random.shuffle(hand_1)
        
        self.hands[0] = hand_0
        self.hands[1] = hand_1
        
        # Random starting player (0 or 1)
        self.current_player = random.randint(0, 1)
        self.moves_made = 0
    
    def to_play(self) -> int:
        """Return the current player.
        
        Returns:
            0 or 1 indicating which player's turn it is.
        """
        return self.current_player
    
    def reset(self) -> np.ndarray:
        """Reset the game to its initial state.
        
        Returns:
            Initial observation of the game.
        """
        self.board = np.full((3, 3), None, dtype=object)
        self.hands = [[], []]
        self.moves_made = 0
        self._initialize_game()
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute a move and return the new game state.
        
        Args:
            action: Integer encoding the move (card_index * 9 + position).
            
        Returns:
            Tuple of (observation, reward, done):
                - observation: numpy array representing the game state
                - reward: Terminal +1/-1/0, or dense +0.1 per card flipped
                - done: True if the game is over
        """
        card_index = action // 9
        position = action % 9
        row = position // 3
        col = position % 3
        
        # Validate and execute move
        if not self._is_legal_action(action):
            raise ValueError(f"Illegal action: {action}")
        
        # Count opponent's cards before placement (for flip reward calculation)
        pre_flip_count = self._count_player_cards(1 - self.current_player)
        
        # Get the card from current player's hand
        card_name = self.hands[self.current_player].pop(card_index)
        
        # Place the card on the board
        self.board[row, col] = (card_name, self.current_player)
        
        # Check for flips and count how many were flipped
        flipped_count = self._check_and_flip_count(row, col, card_name)
        
        # Update game state
        self.moves_made += 1
        game_over = self.moves_made >= 9
        
        # Calculate reward (dense during game, terminal at end)
        reward = 0
        if game_over:
            # Terminal reward
            winner = self._get_winner()
            if winner == self.current_player:
                reward = 1
            elif winner == 1 - self.current_player:
                reward = -1
            else:
                reward = 0  # Draw
        else:
            # Dense reward: +0.1 per opponent card flipped
            # This provides immediate feedback for good placements
            reward = flipped_count * 0.1
            
            # Small penalty for not flipping (encourages aggressive play)
            if flipped_count == 0:
                reward -= 0.02
        
        # Switch player if game is not over
        if not game_over:
            self.current_player = 1 - self.current_player
        
        return self.get_observation(), reward, game_over
    
    def _is_legal_action(self, action: int) -> bool:
        """Check if an action is legal.
        
        Args:
            action: The action to check.
            
        Returns:
            True if the action is legal, False otherwise.
        """
        card_index = action // 9
        position = action % 9
        row = position // 3
        col = position % 3
        
        # Check if card index is valid
        if card_index >= len(self.hands[self.current_player]):
            return False
        
        # Check if position is empty
        if self.board[row, col] is not None:
            return False
        
        return True
    
    def legal_actions(self) -> List[int]:
        """Return the list of legal actions for the current state.
        
        Optimized version using pre-computed templates and inline checks.
        
        Returns:
            List of legal action integers.
        """
        hand = self.hands[self.current_player]
        hand_size = len(hand)
        legal = []
        
        # Use inline checks instead of calling _is_legal_action for each action
        for card_idx in range(hand_size):
            for pos in range(9):
                row, col = pos // 3, pos % 3
                if self.board[row, col] is None:  # Fast inline check
                    legal.append(card_idx * 9 + pos)
        
        return legal
    
    def _check_and_flip(self, row: int, col: int, placed_card: str) -> None:
        """Check adjacent cards and flip any that lose to the placed card.
        
        Args:
            row: Row where the card was placed.
            col: Column where the card was placed.
            placed_card: Name of the card that was placed.
        """
        placed_values = self.card_values[placed_card]
        
        for direction, (d_row, d_col, our_face, their_face) in self.DIRECTIONS.items():
            new_row = row + d_row
            new_col = col + d_col
            
            # Check if the adjacent position is on the board
            if not (0 <= new_row < 3 and 0 <= new_col < 3):
                continue
            
            # Check if there's an opponent's card there
            adjacent = self.board[new_row, new_col]
            if adjacent is None:
                continue
            
            adjacent_card, adjacent_owner = adjacent
            
            # Skip if it's our own card
            if adjacent_owner == self.current_player:
                continue
            
            # Compare face values
            our_value = placed_values[our_face]
            their_value = self.card_values[adjacent_card][their_face]
            
            # Flip if our value is higher
            if our_value > their_value:
                self.board[new_row, new_col] = (adjacent_card, self.current_player)
    
    def _count_player_cards(self, player: int) -> int:
        """Count how many cards a player currently owns on the board.
        
        Args:
            player: The player index (0 or 1).
            
        Returns:
            Number of cards owned by the player.
        """
        count = 0
        for row in range(3):
            for col in range(3):
                if self.board[row, col] is not None:
                    _, owner = self.board[row, col]
                    if owner == player:
                        count += 1
        return count
    
    def _check_and_flip_count(self, row: int, col: int, placed_card: str) -> int:
        """Check adjacent cards, flip any that lose, and return flip count.
        
        Args:
            row: Row where the card was placed.
            col: Column where the card was placed.
            placed_card: Name of the card that was placed.
            
        Returns:
            Number of opponent cards flipped.
        """
        placed_values = self.card_values[placed_card]
        flip_count = 0
        
        for direction, (d_row, d_col, our_face, their_face) in self.DIRECTIONS.items():
            new_row = row + d_row
            new_col = col + d_col
            
            # Check if the adjacent position is on the board
            if not (0 <= new_row < 3 and 0 <= new_col < 3):
                continue
            
            # Check if there's an opponent's card there
            adjacent = self.board[new_row, new_col]
            if adjacent is None:
                continue
            
            adjacent_card, adjacent_owner = adjacent
            
            # Skip if it's our own card
            if adjacent_owner == self.current_player:
                continue
            
            # Compare face values
            our_value = placed_values[our_face]
            their_value = self.card_values[adjacent_card][their_face]
            
            # Flip if our value is higher
            if our_value > their_value:
                self.board[new_row, new_col] = (adjacent_card, self.current_player)
                flip_count += 1
        
        return flip_count
    
    def _get_winner(self) -> Optional[int]:
        """Determine the winner of the game.
        
        Returns:
            0 if player 0 wins, 1 if player 1 wins, None if draw.
        """
        player_0_cards = 0
        player_1_cards = 0
        
        for row in range(3):
            for col in range(3):
                if self.board[row, col] is not None:
                    _, owner = self.board[row, col]
                    if owner == 0:
                        player_0_cards += 1
                    else:
                        player_1_cards += 1
        
        if player_0_cards > player_1_cards:
            return 0
        elif player_1_cards > player_0_cards:
            return 1
        else:
            return None  # Draw
    
    def have_winner(self) -> bool:
        """Check if the game has ended.
        
        Returns:
            True if the game is over, False otherwise.
        """
        return self.moves_made >= 9
    
    def get_observation(self) -> np.ndarray:
        """Generate the current game observation.
        
        The observation is a 3x3x11 numpy array with channels:
        - Channel 0: Player 0's cards (1 if owned by player 0, else 0)
        - Channel 1: Player 1's cards (1 if owned by player 1, else 0)
        - Channel 2: Current player indicator (1 if player 0's turn, -1 if player 1's turn)
        - Channel 3: North values for cards at each position (0 if empty)
        - Channel 4: East values for cards at each position (0 if empty)
        - Channel 5: South values for cards at each position (0 if empty)
        - Channel 6: West values for cards at each position (0 if empty)
        - Channels 7-10: Reserved for future use (currently 0)
        
        Returns:
            Numpy array representing the game observation.
        """
        observation = np.zeros((3, 3, 11), dtype=np.float32)
        
        for row in range(3):
            for col in range(3):
                cell = self.board[row, col]
                if cell is not None:
                    card_name, owner = cell
                    values = self.card_values[card_name]
                    
                    # Player ownership channels
                    if owner == 0:
                        observation[row, col, 0] = 1
                    else:
                        observation[row, col, 1] = 1
                    
                    # Card value channels (N, E, S, W)
                    observation[row, col, 3] = values[0]  # North
                    observation[row, col, 4] = values[1]  # East
                    observation[row, col, 5] = values[2]  # South
                    observation[row, col, 6] = values[3]  # West
        
        # Current player channel
        if self.current_player == 0:
            observation[:, :, 2] = 1
        else:
            observation[:, :, 2] = -1
        
        return observation
    
    def get_board_state(self) -> np.ndarray:
        """Get a simplified board state representation.
        
        Returns:
            3x3 array where each cell contains:
                - None for empty
                - (card_name, owner) tuple for occupied
        """
        return self.board.copy()
    
    def get_hand(self, player: int) -> List[str]:
        """Get the current hand of a player.
        
        Args:
            player: Player index (0 or 1).
            
        Returns:
            List of card names in the player's hand.
        """
        return self.hands[player].copy()
    
    def render(self) -> None:
        """Display the current game state."""
        print("\n=== Triple Triad ===")
        print(f"Moves made: {self.moves_made}/9")
        print(f"Current player: Player {self.current_player + 1}")
        
        print("\nPlayer 1 hand:")
        for i, card in enumerate(self.hands[0]):
            values = self.card_values[card]
            print(f"  {i}: {card} {values}")
        
        print("\nPlayer 2 hand:")
        for i, card in enumerate(self.hands[1]):
            values = self.card_values[card]
            print(f"  {i}: {card} {values}")
        
        print("\nBoard:")
        print("  0   1   2")
        for row in range(3):
            row_str = f"{row} "
            for col in range(3):
                cell = self.board[row, col]
                if cell is None:
                    row_str += " .  "
                else:
                    card_name, owner = cell
                    owner_marker = "1" if owner == 0 else "2"
                    short_name = card_name[:3]
                    row_str += f"{owner_marker}:{short_name} "
            print(row_str)
        
        print()
    
    def expert_action(self) -> int:
        """Return an action from a basic expert agent.
        
        This is a simple heuristic that:
        1. Prioritizes flips
        2. Prefers center position
        3. Prefers corners
        
        Returns:
            The selected action.
        """
        legal_actions = self.legal_actions()
        if not legal_actions:
            return None
        
        best_action = legal_actions[0]
        best_score = -float('inf')
        
        for action in legal_actions:
            score = self._evaluate_action(action)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_action(self, action: int) -> float:
        """Evaluate an action with a simple heuristic.
        
        Args:
            action: The action to evaluate.
            
        Returns:
            Score for the action (higher is better).
        """
        card_index = action // 9
        position = action % 9
        row = position // 3
        col = position % 3
        
        card_name = self.hands[self.current_player][card_index]
        card_values = self.card_values[card_name]
        
        score = 0
        
        # Position bonuses
        if row == 1 and col == 1:  # Center
            score += 10
        elif (row in [0, 2]) and (col in [0, 2]):  # Corners
            score += 5
        
        # Flip potential
        for direction, (d_row, d_col, our_face, their_face) in self.DIRECTIONS.items():
            new_row = row + d_row
            new_col = col + d_col
            
            if not (0 <= new_row < 3 and 0 <= new_col < 3):
                continue
            
            adjacent = self.board[new_row, new_col]
            if adjacent is not None:
                adjacent_card, adjacent_owner = adjacent
                if adjacent_owner != self.current_player:
                    our_value = card_values[our_face]
                    their_value = self.card_values[adjacent_card][their_face]
                    if our_value > their_value:
                        score += 20  # High bonus for guaranteed flip
        
        return score
    
    def get_config(self) -> Dict[str, Any]:
        """Get game configuration for MuZero.
        
        Returns:
            Dictionary containing game configuration parameters.
        """
        return {
            'observation_shape': (3, 3, 11),
            'action_space': list(range(45)),
            'players': [0, 1],
            'max_moves': 9,
        }


# ============== GAME WRAPPER ==============
from typing import Tuple, List, Optional
import numpy as np



class Game(AbstractGame):
    """Game wrapper for Triple Triad.
    
    This class wraps the TripleTriad core logic to provide the interface
    expected by MuZero, including reward scaling and human input handling.
    
    Attributes:
        env: The underlying TripleTriad game instance.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the game wrapper.
        
        Args:
            seed: Optional random seed for reproducible game initialization.
        """
        self.env = TripleTriad(seed=seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply an action to the game.
        
        Args:
            action: Action from the action space.
            
        Returns:
            Tuple of (observation, reward, done).
        """
        observation, reward, done = self.env.step(action)
        # Scale reward by 5 (reduced from 20 to accommodate dense flip rewards)
        # Terminal rewards: +1/-1/0 -> +5/-5/0
        # Dense rewards: +0.1 per flip -> +0.5, -0.02 no-flip -> -0.1
        return observation, reward * 5, done
    
    def to_play(self) -> int:
        """Return the current player.
        
        Returns:
            The current player (0 or 1).
        """
        return self.env.to_play()
    
    def legal_actions(self) -> List[int]:
        """Return the legal actions for the current state.
        
        Returns:
            List of legal action integers.
        """
        return self.env.legal_actions()
    
    def reset(self) -> np.ndarray:
        """Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()
    
    def render(self) -> None:
        """Display the game observation."""
        self.env.render()
        input("Press enter to take a step ")
    
    def human_to_action(self) -> int:
        """Convert human input to an action.
        
        Returns:
            The action selected by the human player.
        """
        while True:
            try:
                hand = self.env.get_hand(self.env.current_player)
                print(f"\nYour hand (Player {self.env.current_player + 1}):")
                for i, card in enumerate(hand):
                    values = self.env.card_values[card]
                    print(f"  {i}: {card} (N:{values[0]}, E:{values[1]}, S:{values[2]}, W:{values[3]})")
                
                card_index = int(input(f"Enter card index (0-{len(hand)-1}): "))
                if card_index < 0 or card_index >= len(hand):
                    print(f"Invalid card index. Must be between 0 and {len(hand)-1}")
                    continue
                
                print("\nBoard positions:")
                print("  0 1 2")
                print("  3 4 5")
                print("  6 7 8")
                
                position = int(input("Enter position (0-8): "))
                if position < 0 or position > 8:
                    print("Invalid position. Must be between 0 and 8")
                    continue
                
                action = card_index * 9 + position
                if action in self.env.legal_actions():
                    return action
                else:
                    print("That action is not legal. Try again.")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def expert_agent(self) -> int:
        """Return an action from the expert agent.
        
        Returns:
            The action selected by the expert agent.
        """
        return self.env.expert_action()
    
    def action_to_string(self, action_number: int) -> str:
        """Convert an action number to a string.
        
        Args:
            action_number: The action number.
            
        Returns:
            String representation of the action.
        """
        card_index = action_number // 9
        position = action_number % 9
        row = position // 3
        col = position % 3
        
        hand = self.env.get_hand(self.env.current_player)
        if card_index < len(hand):
            card_name = hand[card_index]
        else:
            card_name = "Unknown"
        
        return f"Play {card_name} at row {row}, column {col}"
    
    def get_winner(self) -> Optional[int]:
        """Get the winner of the game.
        
        Returns:
            0 if player 0 wins, 1 if player 1 wins, None if draw.
        """
        return self.env._get_winner()
    
    def get_board_state(self) -> np.ndarray:
        """Get the current board state.
        
        Returns:
            The board state array.
        """
        return self.env.get_board_state()
    
    def get_hand(self, player: int) -> list:
        """Get a player's hand.
        
        Args:
            player: The player index (0 or 1).
            
        Returns:
            The player's hand.
        """
        return self.env.get_hand(player)
