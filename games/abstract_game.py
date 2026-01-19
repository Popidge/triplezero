"""Abstract base class for MuZero games."""
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
