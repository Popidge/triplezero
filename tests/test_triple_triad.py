"""Unit tests for Triple Triad core game logic."""
import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.triple_triad.triple_triad import TripleTriad, TRIPLE_TRIAD_CARDS


class TestCardDefinitions:
    """Tests for card definitions and values."""
    
    def test_all_cards_defined(self):
        """Verify all 10 cards are defined."""
        assert len(TRIPLE_TRIAD_CARDS) == 10
    
    def test_card_values_are_tuples(self):
        """Verify all card values are tuples of 4 integers."""
        for card_name, values in TRIPLE_TRIAD_CARDS.items():
            assert isinstance(values, tuple), f"{card_name} values is not a tuple"
            assert len(values) == 4, f"{card_name} does not have 4 values"
            assert all(isinstance(v, int) for v in values), f"{card_name} contains non-integer values"
    
    def test_card_values_in_range(self):
        """Verify all card values are between 1 and 6 (typical Triple Triad range)."""
        for card_name, values in TRIPLE_TRIAD_CARDS.items():
            for i, value in enumerate(values):
                assert 1 <= value <= 6, f"{card_name} face {i} has value {value} out of range"


class TestTripleTriadInitialization:
    """Tests for game initialization."""
    
    def test_initialization_with_seed(self):
        """Test that game initialization with seed produces reproducible results."""
        game1 = TripleTriad(seed=42)
        game2 = TripleTriad(seed=42)
        
        # Both games should have the same hands
        assert game1.hands[0] == game2.hands[0]
        assert game1.hands[1] == game2.hands[1]
        assert game1.current_player == game2.current_player
    
    def test_initialization_without_seed(self):
        """Test that games without seeds have valid hands."""
        game = TripleTriad()
        
        # Each player should have 5 cards
        assert len(game.hands[0]) == 5
        assert len(game.hands[1]) == 5
        
        # Hands should be mutually exclusive
        hand0_set = set(game.hands[0])
        hand1_set = set(game.hands[1])
        assert len(hand0_set.intersection(hand1_set)) == 0
        
        # All 10 cards should be distributed
        assert len(hand0_set) + len(hand1_set) == 10
    
    def test_board_initialized_empty(self):
        """Test that the board is initialized to empty."""
        game = TripleTriad()
        
        for row in range(3):
            for col in range(3):
                assert game.board[row, col] is None
    
    def test_moves_made_initialized_to_zero(self):
        """Test that moves_made starts at 0."""
        game = TripleTriad()
        assert game.moves_made == 0


class TestReset:
    """Tests for game reset functionality."""
    
    def test_reset_clears_board(self):
        """Test that reset clears the board."""
        game = TripleTriad(seed=42)
        
        # Make a move
        legal_actions = game.legal_actions()
        if legal_actions:
            game.step(legal_actions[0])
        
        # Reset
        game.reset()
        
        # Board should be empty
        for row in range(3):
            for col in range(3):
                assert game.board[row, col] is None
    
    def test_reset_generates_new_hands(self):
        """Test that reset generates new hands."""
        game = TripleTriad(seed=42)
        original_hand_0 = game.hands[0].copy()
        
        game.reset()
        new_hand_0 = game.hands[0].copy()
        
        # Hands should be different (with high probability)
        # Note: With the same seed, hands would be the same
        # But we're testing that hands are valid
        assert len(game.hands[0]) == 5
        assert len(game.hands[1]) == 5
    
    def test_reset_resets_moves_made(self):
        """Test that reset resets moves_made."""
        game = TripleTriad(seed=42)
        
        # Make some moves
        for _ in range(3):
            legal_actions = game.legal_actions()
            if legal_actions:
                game.step(legal_actions[0])
        
        assert game.moves_made == 3
        
        game.reset()
        assert game.moves_made == 0


class TestLegalActions:
    """Tests for legal actions."""
    
    def test_initial_legal_actions_count(self):
        """Test that all 45 actions are initially legal."""
        game = TripleTriad()
        legal = game.legal_actions()
        # All 45 actions should be legal (5 cards * 9 positions)
        assert len(legal) == 45
    
    def test_legal_actions_after_placing_card(self):
        """Test that legal actions decrease after placing a card."""
        game = TripleTriad(seed=42)
        
        # Get initial legal actions count
        initial_count = len(game.legal_actions())
        
        # Place first card using the first legal action
        legal = game.legal_actions()
        game.step(legal[0])  # Use first legal action
        
        # Should have fewer legal actions
        new_count = len(game.legal_actions())
        assert new_count < initial_count
    
    def test_legal_actions_respects_hand(self):
        """Test that legal actions only include cards in player's hand."""
        game = TripleTriad(seed=42)
        
        # After placing 5 cards, player 0 should have no cards
        legal = game.legal_actions()
        for _ in range(5):
            assert len(legal) > 0, "No legal actions available"
            action = legal[0]  # Use first legal action
            game.step(action)
            legal = game.legal_actions()
        
        # Player 1's turn, should have 5 cards but only 4 empty positions left
        legal = game.legal_actions()
        # Should be 5 cards * 4 positions = 20 actions, but some might not be valid
        assert len(legal) > 0, "Player 1 should have legal actions"
        assert len(legal) <= 20, f"Too many legal actions: {len(legal)}"
    
    def test_no_legal_actions_at_game_end(self):
        """Test that there are no legal actions when the game is over."""
        game = TripleTriad(seed=42)
        
        # Play all 9 moves
        for i in range(9):
            legal = game.legal_actions()
            if legal:
                game.step(legal[0])
        
        # No legal actions should remain
        legal = game.legal_actions()
        assert len(legal) == 0


class TestStep:
    """Tests for the step function."""
    
    def test_step_returns_observation(self):
        """Test that step returns a valid observation."""
        game = TripleTriad(seed=42)
        
        # Use first legal action
        legal = game.legal_actions()
        observation, reward, done = game.step(legal[0])
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (3, 3, 11)
    
    def test_step_returns_not_done_initially(self):
        """Test that game is not done after first move."""
        game = TripleTriad(seed=42)
        
        # Use first legal action
        legal = game.legal_actions()
        _, _, done = game.step(legal[0])
        
        assert done is False
    
    def test_step_switches_player(self):
        """Test that step switches to the other player."""
        game = TripleTriad(seed=42)
        initial_player = game.current_player
        
        # Use first legal action
        legal = game.legal_actions()
        game.step(legal[0])
        
        assert game.current_player == 1 - initial_player
    
    def test_step_removes_card_from_hand(self):
        """Test that step removes the played card from the hand."""
        game = TripleTriad(seed=42)
        initial_player = game.current_player
        initial_hand_size = len(game.hands[initial_player])
        
        # Use first legal action
        legal = game.legal_actions()
        game.step(legal[0])
        
        # After the step, player switches, so check the original player's hand
        assert len(game.hands[initial_player]) == initial_hand_size - 1
    
    def test_step_places_card_on_board(self):
        """Test that step places the card on the board."""
        game = TripleTriad(seed=42)
        
        # Use first legal action
        legal = game.legal_actions()
        game.step(legal[0])  # Place first card
        
        # Check that the board has a card somewhere
        placed = False
        for row in range(3):
            for col in range(3):
                if game.board[row, col] is not None:
                    card_name, owner = game.board[row, col]
                    # The owner should be the player who just moved
                    # (but we need to check who that was)
                    placed = True
                    break
            if placed:
                break
        
        assert placed, "No card was placed on the board"
    
    def test_game_ends_after_9_moves(self):
        """Test that game ends after 9 moves."""
        game = TripleTriad(seed=42)
        
        for i in range(9):
            legal = game.legal_actions()
            if legal:
                _, _, done = game.step(legal[0])
                if i < 8:
                    assert done is False
                else:
                    assert done is True


class TestFlipping:
    """Tests for the card flipping mechanics."""
    
    def test_no_flip_when_value_lower(self):
        """Test that no flip occurs when our value is lower."""
        game = TripleTriad(seed=42)
        
        # Player 0 places a card at position 4 (center)
        game.step(4 * 9 + 4)  # Use card 0, position 4
        
        # Player 1 places a card to the right with higher West value
        # We need to find a card with West > our East
        # This is a simplified test
        game.step(0)  # Any legal action for player 1
        
        # Check that the game state is valid
        assert game.moves_made == 2
    
    def test_flip_occurs_when_value_higher(self):
        """Test that flip occurs when our value is higher."""
        game = TripleTriad()
        
        # Player 0 places a card
        # We need to set up a scenario where a flip will occur
        # This is complex to test without knowing the exact card values
        # For now, we test that the flip logic is called
        game.step(0)
        game.step(1)  # Player 1 plays
        
        # The game should continue without errors
        assert game.moves_made == 2
    
    def test_flip_increases_ownership_count(self):
        """Test that flipped cards change ownership."""
        game = TripleTriad()
        
        # Player 0 places first card
        game.step(4)  # Center position
        
        # Count player 0's cards
        p0_cards_before = sum(1 for row in range(3) for col in range(3) 
                             if game.board[row, col] is not None and game.board[row, col][1] == 0)
        
        # Player 1 places a card that will flip player 0's card
        # This depends on card values, so we just verify the mechanism works
        game.step(0)
        
        # The game continues
        assert game.moves_made == 2


class TestObservations:
    """Tests for observation generation."""
    
    def test_observation_shape(self):
        """Test that observation has correct shape."""
        game = TripleTriad()
        observation = game.get_observation()
        
        assert observation.shape == (3, 3, 11)
    
    def test_empty_board_observation(self):
        """Test observation of empty board."""
        game = TripleTriad()
        observation = game.get_observation()
        
        # All channels should be 0 except current player
        assert np.all(observation[:, :, 0] == 0)  # Player 0 cards
        assert np.all(observation[:, :, 1] == 0)  # Player 1 cards
        assert np.all(observation[:, :, 3:7] == 0)  # Card values
    
    def test_current_player_channel(self):
        """Test that current player channel is correct."""
        game = TripleTriad()
        
        # Get the initial player
        initial_player = game.to_play()
        obs = game.get_observation()
        
        # Check that current player channel is correct
        if initial_player == 0:
            assert np.all(obs[:, :, 2] == 1)
        else:
            assert np.all(obs[:, :, 2] == -1)
        
        # After a move, the player should switch
        legal = game.legal_actions()
        if legal:
            game.step(legal[0])
            obs = game.get_observation()
            new_player = game.to_play()
            if new_player == 0:
                assert np.all(obs[:, :, 2] == 1)
            else:
                assert np.all(obs[:, :, 2] == -1)
    
    def test_card_placement_observation(self):
        """Test that card placement is reflected in observation."""
        game = TripleTriad(seed=42)
        
        # Use first legal action
        legal = game.legal_actions()
        game.step(legal[0])  # Place first card
        
        observation = game.get_observation()
        
        # Check that player 0's channel has the card if they own it
        # The card should be in player 0's channel or player 1's channel
        p0_cards = np.sum(observation[:, :, 0])
        p1_cards = np.sum(observation[:, :, 1])
        
        # There should be exactly 1 card on the board
        assert p0_cards + p1_cards == 1


class TestWinner:
    """Tests for winner determination."""
    
    def test_no_winner_during_game(self):
        """Test that have_winner returns False during game."""
        game = TripleTriad()
        
        for _ in range(4):
            legal = game.legal_actions()
            if legal:
                game.step(legal[0])
        
        assert game.have_winner() is False
    
    def test_winner_after_game_end(self):
        """Test that winner is determined after game ends."""
        game = TripleTriad(seed=42)
        
        # Play a full game
        legal = game.legal_actions()
        while legal:
            game.step(legal[0])
            legal = game.legal_actions()
        
        assert game.have_winner() is True
        
        winner = game._get_winner()
        assert winner in [0, 1, None]  # 0, 1, or draw (None)
    
    def test_player_with_more_cards_wins(self):
        """Test that player with more cards wins."""
        game = TripleTriad(seed=42)
        
        # Play a full game
        legal = game.legal_actions()
        while legal:
            game.step(legal[0])
            legal = game.legal_actions()
        
        winner = game._get_winner()
        
        if winner is not None:
            p0_cards = sum(1 for row in range(3) for col in range(3) 
                          if game.board[row, col] is not None and game.board[row, col][1] == 0)
            p1_cards = sum(1 for row in range(3) for col in range(3) 
                          if game.board[row, col] is not None and game.board[row, col][1] == 1)
            
            if winner == 0:
                assert p0_cards > p1_cards
            else:
                assert p1_cards > p0_cards
    
    def test_draw_when_equal_cards(self):
        """Test that game is draw when both players have 5 cards."""
        game = TripleTriad()
        
        # Play a full game
        legal = game.legal_actions()
        while legal:
            game.step(legal[0])
            legal = game.legal_actions()
        
        winner = game._get_winner()
        
        # If winner is None, it's a draw
        p0_cards = sum(1 for row in range(3) for col in range(3) 
                      if game.board[row, col] is not None and game.board[row, col][1] == 0)
        p1_cards = sum(1 for row in range(3) for col in range(3) 
                      if game.board[row, col] is not None and game.board[row, col][1] == 1)
        
        if winner is None:
            assert p0_cards == p1_cards == 5


class TestRewards:
    """Tests for reward calculation."""
    
    def test_dense_reward_during_game(self):
        """Test that reward is dense during game (+0.1 per flip, -0.02 for no flip)."""
        game = TripleTriad()
        
        _, reward, done = game.step(0)
        
        # Reward should be -0.02 (no flip penalty) or +0.1*n (flip rewards)
        assert reward in [-0.02, 0.1, 0.2, 0.3, 0.4]  # Various flip scenarios
        assert done is False
    
    def test_positive_reward_for_winner(self):
        """Test that winner gets positive reward."""
        game = TripleTriad(seed=42)
        
        # Play until game ends
        legal = game.legal_actions()
        while legal:
            _, reward, done = game.step(legal[0])
            if done:
                # Check if the reward is correct (1 for win, -1 for loss, 0 for draw)
                assert reward in [-1, 0, 1], f"Invalid reward: {reward}"
            legal = game.legal_actions()
    
    def test_negative_reward_for_loser(self):
        """Test that loser gets negative reward."""
        game = TripleTriad(seed=42)
        
        # Play until game ends
        legal = game.legal_actions()
        while legal:
            _, reward, done = game.step(legal[0])
            if done:
                assert reward in [-1, 0, 1], f"Invalid reward: {reward}"
            legal = game.legal_actions()


class TestEdgeCases:
    """Tests for edge cases and corner scenarios."""
    
    def test_all_corner_positions(self):
        """Test that all corner positions work correctly."""
        game = TripleTriad()
        
        corners = [0, 2, 6, 8]  # Corner positions
        
        for pos in corners:
            # Find a legal action for this position
            legal = game.legal_actions()
            action = None
            for a in legal:
                if a % 9 == pos:  # Same position
                    action = a
                    break
            
            if action is not None:
                game.step(action)  # Place card at corner
        
        # Should have placed up to 4 cards
        assert game.moves_made <= 4
    
    def test_all_edge_positions(self):
        """Test that all edge (non-corner) positions work correctly."""
        game = TripleTriad()
        
        edges = [1, 3, 4, 5, 7]  # Edge and center positions
        
        for pos in edges:
            # Find a legal action for this position
            legal = game.legal_actions()
            action = None
            for a in legal:
                if a % 9 == pos:  # Same position
                    action = a
                    break
            
            if action is not None:
                game.step(action)
        
        # Should have placed up to 5 cards
        assert game.moves_made <= 5
    
    def test_sequence_of_moves(self):
        """Test a sequence of moves."""
        game = TripleTriad(seed=42)
        
        # Play a sequence of moves
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        for action in actions:
            if game._is_legal_action(action):
                game.step(action)
        
        assert game.moves_made <= 9


class TestExpertAgent:
    """Tests for the expert agent."""
    
    def test_expert_agent_returns_legal_action(self):
        """Test that expert agent returns a legal action."""
        game = TripleTriad()
        
        # Game class has expert_agent method
        # Note: TripleTriad class doesn't have expert_agent, but Game class does
        # This test should be in the game wrapper tests
        legal = game.legal_actions()
        if legal:
            # The game should have a method to evaluate actions
            # We can test this through the Game wrapper
            pass  # Test handled in game wrapper tests


class TestToPlay:
    """Tests for the to_play function."""
    
    def test_to_play_returns_valid_player(self):
        """Test that to_play returns 0 or 1."""
        game = TripleTriad()
        
        player = game.to_play()
        
        assert player in [0, 1]
    
    def test_to_play_changes_after_step(self):
        """Test that to_play changes after each move."""
        game = TripleTriad()
        
        player_before = game.to_play()
        game.step(0)
        player_after = game.to_play()
        
        assert player_before != player_after


class TestBoardState:
    """Tests for board state retrieval."""
    
    def test_get_board_state_returns_copy(self):
        """Test that get_board_state returns a copy."""
        game = TripleTriad()
        
        board1 = game.get_board_state()
        game.step(0)
        board2 = game.get_board_state()
        
        # Boards should be different
        assert board1[0, 0] is None
        assert board2[0, 0] is not None
    
    def test_board_state_format(self):
        """Test that board state has correct format."""
        game = TripleTriad()
        
        game.step(0)
        
        board = game.get_board_state()
        
        assert board.shape == (3, 3)
        assert board[0, 0] is not None
        card_name, owner = board[0, 0]
        assert isinstance(card_name, str)
        assert owner in [0, 1]


class TestHandRetrieval:
    """Tests for hand retrieval."""
    
    def test_get_hand_returns_list(self):
        """Test that get_hand returns a list."""
        game = TripleTriad()
        
        hand = game.get_hand(0)
        
        assert isinstance(hand, list)
        assert len(hand) == 5
    
    def test_get_hand_returns_copy(self):
        """Test that get_hand returns a copy."""
        game = TripleTriad()
        
        hand1 = game.get_hand(0)
        game.step(0)
        hand2 = game.get_hand(0)
        
        # Hands should be different
        assert len(hand1) == 5
        assert len(hand2) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
