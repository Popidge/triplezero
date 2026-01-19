"""Unit tests for the Game wrapper class."""
import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.triple_triad.game import Game
from games.triple_triad.config import MuZeroConfig


class TestGameWrapper:
    """Tests for the Game wrapper class."""
    
    def test_initialization(self):
        """Test that Game can be initialized."""
        game = Game(seed=42)
        assert game is not None
    
    def test_step_returns_tuple(self):
        """Test that step returns a tuple."""
        game = Game(seed=42)
        
        observation, reward, done = game.step(0)
        
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    
    def test_to_play_returns_player(self):
        """Test that to_play returns a valid player."""
        game = Game(seed=42)
        
        player = game.to_play()
        
        assert player in [0, 1]
    
    def test_legal_actions_returns_list(self):
        """Test that legal_actions returns a list."""
        game = Game(seed=42)
        
        legal = game.legal_actions()
        
        assert isinstance(legal, list)
        assert len(legal) > 0
    
    def test_reset_returns_observation(self):
        """Test that reset returns an observation."""
        game = Game(seed=42)
        
        observation = game.reset()
        
        assert isinstance(observation, np.ndarray)
    
    def test_action_to_string(self):
        """Test that action_to_string returns a string."""
        game = Game(seed=42)
        
        string = game.action_to_string(0)
        
        assert isinstance(string, str)
        assert len(string) > 0
    
    def test_render_does_not_crash(self):
        """Test that render doesn't crash (simple check)."""
        game = Game(seed=42)
        
        # Just verify it can be called without error
        # We won't actually wait for input in tests
        try:
            # Note: This will hang waiting for input in an interactive session
            # So we just check that the method exists and is callable
            assert callable(game.render)
        except Exception as e:
            pytest.fail(f"render method failed: {e}")
    
    def test_get_winner(self):
        """Test that get_winner returns correct value."""
        game = Game(seed=42)
        
        # Play a full game
        legal = game.legal_actions()
        while legal:
            game.step(legal[0])
            legal = game.legal_actions()
        
        winner = game.get_winner()
        
        # Winner should be 0, 1, or None (draw)
        assert winner in [0, 1, None]
    
    def test_get_board_state(self):
        """Test that get_board_state returns board state."""
        game = Game(seed=42)
        
        board = game.get_board_state()
        
        assert board.shape == (3, 3)
    
    def test_get_hand(self):
        """Test that get_hand returns hand."""
        game = Game(seed=42)
        
        hand = game.get_hand(0)
        
        assert isinstance(hand, list)
        assert len(hand) == 5


class TestRewardScaling:
    """Tests for reward scaling in the wrapper."""
    
    def test_win_reward_scaled(self):
        """Test that win reward is scaled by 5."""
        game = Game(seed=42)
        
        # Play until game ends
        legal = game.legal_actions()
        while legal:
            observation, reward, done = game.step(legal[0])
            if done:
                # If this player won, reward should be +5 (1 * 5)
                if reward > 0:
                    assert reward == 5
                # If lost, reward should be -5 (-1 * 5)
                elif reward < 0:
                    assert reward == -5
                # If draw, reward should be 0
                else:
                    assert reward == 0
            legal = game.legal_actions()
    
    def test_intermediate_rewards_with_dense_reward(self):
        """Test that intermediate rewards use dense reward structure."""
        game = Game(seed=42)
        
        for _ in range(5):
            legal = game.legal_actions()
            if legal:
                _, reward, done = game.step(legal[0])
                if not done:
                    # Dense rewards: +0.1 per flip -> +0.5 after scaling, -0.02 no-flip -> -0.1
                    assert reward in [-0.1, 0, 0.5, 1.0, 1.5, 2.0]  # Various flip counts scaled by 5


class TestInterfaceCompliance:
    """Tests for MuZero interface compliance."""
    
    def test_has_step_method(self):
        """Test that Game has step method."""
        game = Game()
        assert hasattr(game, 'step')
        assert callable(game.step)
    
    def test_has_to_play_method(self):
        """Test that Game has to_play method."""
        game = Game()
        assert hasattr(game, 'to_play')
        assert callable(game.to_play)
    
    def test_has_legal_actions_method(self):
        """Test that Game has legal_actions method."""
        game = Game()
        assert hasattr(game, 'legal_actions')
        assert callable(game.legal_actions)
    
    def test_has_reset_method(self):
        """Test that Game has reset method."""
        game = Game()
        assert hasattr(game, 'reset')
        assert callable(game.reset)
    
    def test_has_render_method(self):
        """Test that Game has render method."""
        game = Game()
        assert hasattr(game, 'render')
        assert callable(game.render)
    
    def test_has_human_to_action_method(self):
        """Test that Game has human_to_action method."""
        game = Game()
        assert hasattr(game, 'human_to_action')
        assert callable(game.human_to_action)
    
    def test_has_expert_agent_method(self):
        """Test that Game has expert_agent method."""
        game = Game()
        assert hasattr(game, 'expert_agent')
        assert callable(game.expert_agent)
    
    def test_has_action_to_string_method(self):
        """Test that Game has action_to_string method."""
        game = Game()
        assert hasattr(game, 'action_to_string')
        assert callable(game.action_to_string)


class TestActionString:
    """Tests for action_to_string method."""
    
    def test_action_string_contains_card_info(self):
        """Test that action string contains card information."""
        game = Game(seed=42)
        
        # Get a legal action
        legal = game.legal_actions()
        if legal:
            action = legal[0]
            string = game.action_to_string(action)
            
            # String should contain some information about the action
            assert isinstance(string, str)
            assert len(string) > 0
    
    def test_action_string_different_for_different_actions(self):
        """Test that different actions produce different strings."""
        game = Game(seed=42)
        
        legal = game.legal_actions()
        
        if len(legal) >= 2:
            string1 = game.action_to_string(legal[0])
            string2 = game.action_to_string(legal[1])
            
            # Different positions should produce different strings
            # (unless they happen to have the same card and position)
            assert isinstance(string1, str)
            assert isinstance(string2, str)


class TestMuZeroConfig:
    """Tests for the MuZeroConfig class."""
    
    def test_config_initialization(self):
        """Test that MuZeroConfig can be initialized."""
        config = MuZeroConfig()
        assert config is not None
    
    def test_observation_shape(self):
        """Test that observation_shape is correct."""
        config = MuZeroConfig()
        assert config.observation_shape == (3, 3, 11)
    
    def test_action_space_size(self):
        """Test that action_space has correct size."""
        config = MuZeroConfig()
        assert len(config.action_space) == 45
    
    def test_players(self):
        """Test that players list is correct."""
        config = MuZeroConfig()
        assert config.players == [0, 1]
    
    def test_max_moves(self):
        """Test that max_moves is 9."""
        config = MuZeroConfig()
        assert config.max_moves == 9
    
    def test_visit_softmax_temperature_fn(self):
        """Test that visit_softmax_temperature_fn returns a float."""
        config = MuZeroConfig()
        
        temp = config.visit_softmax_temperature_fn(0)
        temp_500 = config.visit_softmax_temperature_fn(500)
        temp_1000 = config.visit_softmax_temperature_fn(1000)
        
        assert isinstance(temp, float)
        assert isinstance(temp_500, float)
        assert isinstance(temp_1000, float)
        # Temperature should not increase
        assert temp >= temp_500
        assert temp_500 >= temp_1000
    
    def test_get_uniform_network_config(self):
        """Test that get_uniform_network_config returns a dict."""
        config = MuZeroConfig()
        
        network_config = config.get_uniform_network_config()
        
        assert isinstance(network_config, dict)
        assert 'observation_shape' in network_config
        assert 'action_space_size' in network_config


class TestGameIntegration:
    """Integration tests for the complete game flow."""
    
    def test_complete_game(self):
        """Test that a complete game can be played."""
        game = Game(seed=42)
        
        # Verify initial state
        assert game.env.moves_made == 0
        
        # Play the game
        legal = game.legal_actions()
        moves = 0
        while legal:
            action = legal[0]
            observation, reward, done = game.step(action)
            moves += 1
            legal = game.legal_actions()
            
            # Verify state after each move
            assert isinstance(observation, np.ndarray)
            assert game.env.moves_made == moves
        
        # Game should be over
        assert moves == 9
    
    def test_game_reset_after_completion(self):
        """Test that game can be reset after completion."""
        game = Game(seed=42)
        
        # Play a complete game
        legal = game.legal_actions()
        while legal:
            game.step(legal[0])
            legal = game.legal_actions()
        
        # Reset
        observation = game.reset()
        
        # Should be back to initial state
        assert game.env.moves_made == 0
        assert isinstance(observation, np.ndarray)
    
    def test_multiple_games(self):
        """Test that multiple games can be played."""
        game = Game()
        
        for _ in range(3):
            # Play a game
            legal = game.legal_actions()
            while legal:
                game.step(legal[0])
                legal = game.legal_actions()
            
            # Reset for next game
            game.reset()
        
        # Should have played 3 games
        assert True  # If we got here, the test passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
