"""Game wrapper for MuZero compatibility."""
from typing import Tuple, List, Optional
import numpy as np

from games.abstract_game import AbstractGame
from games.triple_triad.triple_triad import TripleTriad


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
