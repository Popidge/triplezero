"""Triple Triad core game logic implementation."""
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import random


# Card definitions: name -> (North, East, South, West) values
TRIPLE_TRIAD_CARDS = {
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
}


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
        
        # Game state
        self.board: np.ndarray = np.full((3, 3), None, dtype=object)
        self.hands: List[List[str]] = [[], []]
        self.current_player: int = 0
        self.moves_made: int = 0
        
        # Initialize the game
        self._initialize_game()
    
    def _initialize_game(self) -> None:
        """Deal cards to players and determine starting player."""
        # Shuffle cards and deal 5 to each player
        shuffled_cards = self.card_names.copy()
        random.shuffle(shuffled_cards)
        
        self.hands[0] = shuffled_cards[:5]
        self.hands[1] = shuffled_cards[5:10]
        
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
                - reward: 1 if current player wins, -1 if loses, 0 otherwise
                - done: True if the game is over
        """
        card_index = action // 9
        position = action % 9
        row = position // 3
        col = position % 3
        
        # Validate and execute move
        if not self._is_legal_action(action):
            raise ValueError(f"Illegal action: {action}")
        
        # Get the card from current player's hand
        card_name = self.hands[self.current_player].pop(card_index)
        
        # Place the card on the board
        self.board[row, col] = (card_name, self.current_player)
        
        # Check for flips
        self._check_and_flip(row, col, card_name)
        
        # Update game state
        self.moves_made += 1
        game_over = self.moves_made >= 9
        
        # Calculate reward if game is over
        reward = 0
        if game_over:
            winner = self._get_winner()
            if winner == self.current_player:
                reward = 1
            elif winner == 1 - self.current_player:
                reward = -1
            else:
                reward = 0  # Draw
        
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
        
        Returns:
            List of legal action integers.
        """
        legal = []
        hand = self.hands[self.current_player]
        
        for card_idx in range(len(hand)):
            for pos in range(9):
                action = card_idx * 9 + pos
                if self._is_legal_action(action):
                    legal.append(action)
        
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
        - Channels 3-6: North values for cards at each position
        - Channels 7-10: East values for cards at each position
        - Channels 11-14: South values for cards at each position
        - Channels 15-18: West values for cards at each position
        
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
