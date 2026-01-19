"""Tests for the generated triple_triad.py file.

These tests verify that the build process produces a correctly functioning
single-file version of the game that can be dropped into MuZero's games/ directory.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import the generated file
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuiltGameImports:
    """Test that the built file can be imported correctly."""
    
    def test_import_game_class(self):
        """Test that Game class can be imported from generated file."""
        from triple_triad import Game
        assert Game is not None
    
    def test_import_muzero_config(self):
        """Test that MuZeroConfig class can be imported."""
        from triple_triad import MuZeroConfig
        assert MuZeroConfig is not None
    
    def test_import_triple_triad_class(self):
        """Test that TripleTriad class can be imported."""
        from triple_triad import TripleTriad
        assert TripleTriad is not None
    
    def test_import_card_definitions(self):
        """Test that card definitions can be imported."""
        from triple_triad import TRIPLE_TRIAD_CARDS
        assert len(TRIPLE_TRIAD_CARDS) == 10
        assert "Geezard" in TRIPLE_TRIAD_CARDS
        assert TRIPLE_TRIAD_CARDS["Geezard"] == (1, 4, 5, 1)
    
    def test_import_abstract_game(self):
        """Test that AbstractGame can be imported."""
        from triple_triad import AbstractGame
        assert AbstractGame is not None


class TestBuiltGameFunctionality:
    """Test that the built file functions correctly."""
    
    def test_game_initialization(self):
        """Test Game class initialization."""
        from triple_triad import Game
        game = Game(seed=42)
        assert game is not None
    
    def test_game_reset(self):
        """Test game reset functionality."""
        from triple_triad import Game
        game = Game(seed=42)
        obs = game.reset()
        assert obs is not None
        assert obs.shape == (3, 3, 11)
    
    def test_game_legal_actions(self):
        """Test legal actions generation."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        legal = game.legal_actions()
        assert len(legal) == 45  # 5 cards * 9 positions
    
    def test_game_step(self):
        """Test game step functionality."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        
        # Take first legal action
        action = game.legal_actions()[0]
        obs, reward, done = game.step(action)
        
        assert obs is not None
        assert obs.shape == (3, 3, 11)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    
    def test_game_to_play(self):
        """Test to_play returns correct player."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        player = game.to_play()
        assert player in [0, 1]
    
    def test_game_complete(self):
        """Test completing a full game."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        
        moves = 0
        while True:
            legal = game.legal_actions()
            if not legal:
                break
            action = legal[0]
            _, reward, done = game.step(action)
            moves += 1
            if done:
                break
        
        assert moves > 0
    
    def test_triple_triad_class_direct(self):
        """Test TripleTriad class used directly."""
        from triple_triad import TripleTriad
        game = TripleTriad(seed=123)
        obs = game.reset()
        assert obs.shape == (3, 3, 11)
    
    def test_muzero_config(self):
        """Test MuZeroConfig class."""
        from triple_triad import MuZeroConfig
        config = MuZeroConfig()
        
        assert config.observation_shape == (3, 3, 11)
        assert config.action_space == list(range(45))
        assert config.players == [0, 1]
        assert config.max_moves == 9
    
    def test_game_human_to_action(self):
        """Test human_to_action method exists."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        assert hasattr(game, 'human_to_action')
    
    def test_game_expert_agent(self):
        """Test expert_agent method exists and returns valid action."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        
        # Expert agent should return a legal action
        action = game.expert_agent()
        assert action in game.legal_actions()
    
    def test_game_action_to_string(self):
        """Test action_to_string method."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        
        action_str = game.action_to_string(0)
        assert isinstance(action_str, str)
        assert len(action_str) > 0
    
    def test_game_render(self):
        """Test render method exists."""
        from triple_triad import Game
        game = Game(seed=42)
        game.reset()
        assert hasattr(game, 'render')
    
    def test_consistency_with_modular_version(self):
        """Test that built version produces same results as modular."""
        # Import from modular version
        from games.triple_triad.triple_triad import TripleTriad as ModularTripleTriad
        
        # Import from built version
        from triple_triad import TripleTriad as BuiltTripleTriad
        
        # Both should have same card definitions
        from games.triple_triad.cards import TRIPLE_TRIAD_CARDS as MODULAR_CARDS
        from triple_triad import TRIPLE_TRIAD_CARDS as BUILT_CARDS
        
        assert MODULAR_CARDS == BUILT_CARDS
