# Triple Triad Game Implementation Plan

## Overview
This document outlines the implementation of the Triple Triad minigame from Final Fantasy 8, designed for use with MuZero self-play training.

## Game Rules Summary

### Basic Setup
- **Board**: 3x3 grid (9 positions)
- **Players**: 2 players, each with 5 cards
- **Cards**: 10 unique cards with 4 directional values (North, East, South, West)
- **Ruleset**: "Open" only (perfect information - both players can see all cards)
- **Starting Player**: Randomly determined

### Gameplay Mechanics
1. Players alternate turns placing one card from their hand onto an empty board position
2. When placing a card adjacent to an opponent's card, compare the touching face values
3. If the newly placed card's value is higher, the opponent's card is "flipped" to the current player's ownership
4. Game ends when all 9 positions are filled
5. Winner: Player with the most cards on the board (5-5 is a draw)

### Card Values
Each card has 4 values: (North, East, South, West)
- North = Top face
- East = Right face
- South = Bottom face
- West = Left face

### Flipping Logic
When Player B places a card to the right of Player A's card:
- Compare Player A's East value vs Player B's West value
- If Player B's West > Player A's East, flip Player A's card to Player B's ownership

## Architecture Design

### Project Structure
```
triplezero/
├── games/
│   ├── __init__.py
│   ├── abstract_game.py          # Base class for MuZero games
│   └── triple_triad/
│       ├── __init__.py
│       ├── game.py               # Game wrapper (MuZero interface)
│       ├── triple_triad.py       # Core game logic
│       └── config.py             # MuZeroConfig for Triple Triad
├── tests/
│   ├── __init__.py
│   ├── test_triple_triad.py      # Core game logic tests
│   └── test_game_wrapper.py      # Game wrapper tests
├── cards.txt                     # Card definitions
├── pyproject.toml
└── README.md
```

### Component Responsibilities

#### 1. `TripleTriad` Class (Core Logic)
**Location**: `games/triple_triad/triple_triad.py`

**Responsibilities**:
- Manage game state (board, hands, current player)
- Implement card placement and flipping logic
- Determine legal actions
- Check for game termination
- Calculate rewards
- Generate observations

**Key Methods**:
- `__init__(seed=None)`: Initialize game with random hands
- `reset()`: Reset game to initial state
- `step(action)`: Execute a move and return (observation, reward, done)
- `legal_actions()`: Return list of legal moves
- `to_play()`: Return current player (0 or 1)
- `get_observation()`: Return current observation tensor
- `have_winner()`: Check if game has ended (board full)
- `get_winner()`: Determine winner (0, 1, or None for draw)

**State Representation**:
- `board`: 3x3 array storing card information
  - Each cell: (card_id, owner) or None
- `hands`: List of 2 lists, each containing 5 card IDs
- `current_player`: 0 or 1
- `card_values`: Dictionary mapping card_id to (N, E, S, W) tuple

#### 2. `Game` Class (MuZero Wrapper)
**Location**: `games/triple_triad/game.py`

**Responsibilities**:
- Wrap `TripleTriad` class to match MuZero's `AbstractGame` interface
- Scale rewards appropriately for training
- Provide human-to-action mapping
- Implement expert agent (optional)

**Key Methods**:
- `__init__(seed=None)`: Initialize game
- `step(action)`: Execute action, scale reward
- `to_play()`: Return current player
- `legal_actions()`: Return legal actions
- `reset()`: Reset game
- `render()`: Display game state
- `human_to_action()`: Convert human input to action
- `action_to_string(action)`: Convert action to descriptive string

#### 3. `MuZeroConfig` Class
**Location**: `games/triple_triad/config.py`

**Key Configuration Parameters**:
- `observation_shape`: (3, 3, 11) - board state + card information
  - Channel 0: Player 0's cards on board
  - Channel 1: Player 1's cards on board
  - Channel 2: Current player indicator
  - Channels 3-10: Card values for each position (N, E, S, W for each card)
- `action_space`: List of 45 possible actions (5 cards × 9 positions)
- `players`: [0, 1]
- `max_moves`: 9 (one card per board position)

#### 4. `AbstractGame` Base Class
**Location**: `games/abstract_game.py`

**Purpose**: Define the interface that all MuZero games must implement.

## Action Space Design

### Action Encoding
Each action is a single integer from 0-44 representing:
```
action = card_index * 9 + board_position
```
Where:
- `card_index`: 0-4 (which card from hand to play)
- `board_position`: 0-8 (which board position to place it)

**Example**:
- Action 0: Play card 0 at position 0
- Action 8: Play card 0 at position 8
- Action 9: Play card 1 at position 0
- Action 44: Play card 4 at position 8

### Legal Action Filtering
At each turn, only actions where:
1. The selected card is still in the player's hand
2. The target board position is empty

## Observation Space Design

### Observation Tensor Shape
Shape: `(3, 3, 11)` where:
- **Height**: 3 (board rows)
- **Width**: 3 (board columns)
- **Channels**: 11

### Channel Breakdown
1. **Channel 0**: Player 0's cards (1 if owned by player 0, else 0)
2. **Channel 1**: Player 1's cards (1 if owned by player 1, else 0)
3. **Channel 2**: Current player (1 if player 0's turn, -1 if player 1's turn)
4. **Channels 3-6**: North values for cards at each position
5. **Channels 7-10**: East values for cards at each position
6. **Channels 11-14**: South values for cards at each position
7. **Channels 15-18**: West values for cards at each position

**Note**: For empty positions, all value channels are 0.

## Reward Design

### Reward Structure
- **Win**: +1
- **Loss**: -1
- **Draw**: 0
- **Intermediate moves**: 0 (no reward until game ends)

The `Game` wrapper will scale rewards by 20 (similar to tictactoe) for better training dynamics.

## Unit Test Strategy

### Test Categories

#### 1. Core Game Logic Tests (`test_triple_triad.py`)
- **Card Assignment Tests**:
  - Verify random hand generation
  - Ensure hands are mutually exclusive
  - Test seed reproducibility

- **Board State Tests**:
  - Initial board is empty
  - Card placement updates board correctly
  - Board position tracking

- **Legal Actions Tests**:
  - All 45 actions available at start
  - Actions filtered correctly after card played
  - Actions filtered correctly when position occupied
  - No legal actions when board full

- **Flipping Logic Tests**:
  - Test all 4 directions (N, E, S, W)
  - Verify flipping when new card value > adjacent card value
  - Verify no flipping when new card value ≤ adjacent card value
  - Test multiple flips in single move
  - Test corner cases (multiple adjacent cards)

- **Game Termination Tests**:
  - Game ends after 9 moves
  - Winner calculation (player 0 wins)
  - Winner calculation (player 1 wins)
  - Draw detection (5-5)

- **Observation Tests**:
  - Observation shape is correct
  - Channel values are correct
  - Current player channel updates correctly
  - Card values are encoded correctly

- **Reset Tests**:
  - Reset clears board
  - Reset generates new hands
  - Reset resets current player

#### 2. Game Wrapper Tests (`test_game_wrapper.py`)
- **Interface Compliance Tests**:
  - All required methods implemented
  - Method signatures match `AbstractGame`

- **Reward Scaling Tests**:
  - Win reward scaled to 20
  - Loss reward scaled to -20
  - Draw reward remains 0

- **Action String Tests**:
  - `action_to_string` returns descriptive strings

- **Human Input Tests**:
  - `human_to_action` converts input correctly

### Test Fixtures

#### `test_triple_triad.py`
```python
@pytest.fixture
def game():
    return TripleTriad(seed=42)

@pytest.fixture
def fixed_hands_game():
    """Game with predetermined hands for deterministic testing"""
    return TripleTriad(seed=42)
```

#### `test_game_wrapper.py`
```python
@pytest.fixture
def game_wrapper():
    return Game(seed=42)
```

### Test Data

#### Sample Hands for Testing
Using seed 42 for reproducibility:
- Player 0: [card_ids...]
- Player 1: [card_ids...]

#### Test Scenarios
1. **Simple Flip**: Player places card adjacent to one opponent card
2. **Double Flip**: Player places card between two opponent cards
3. **Triple Flip**: Player places card with three adjacent opponent cards
4. **No Flip**: Player places card with lower values than adjacent cards
5. **Chain Reaction**: Flip that enables subsequent flips (not applicable in basic rules)

## Implementation Order

### Phase 1: Foundation
1. Create directory structure
2. Add dependencies to `pyproject.toml`
3. Create `abstract_game.py` base class
4. Implement `TripleTriad` core logic
5. Create basic unit tests for core logic

### Phase 2: Game Wrapper
1. Implement `Game` wrapper class
2. Implement `MuZeroConfig` class
3. Create tests for wrapper

### Phase 3: Validation
1. Run all tests
2. Fix any bugs
3. Add edge case tests
4. Verify observation encoding
5. Verify action encoding

## Dependencies

### Required Packages
- `numpy`: Array operations and board representation
- `torch`: MuZero framework (for future use)
- `pytest`: Unit testing framework

### Installation
```bash
uv add numpy torch
uv add --dev pytest pytest-cov
```

## Success Criteria

1. ✅ All unit tests pass
2. ✅ Game logic correctly implements Triple Triad rules
3. ✅ Observation encoding is correct and complete
4. ✅ Action space is properly defined
5. ✅ Legal actions are correctly filtered
6. ✅ Game termination detection works
7. ✅ Winner calculation is accurate
8. ✅ Code follows project conventions (uv, type hints, docstrings)

## Future Enhancements

1. **Additional Rulesets**: Implement "Random", "Same", "Plus", "Combo" rules
2. **Expert Agent**: Implement a strong heuristic agent for evaluation
3. **Visualization**: Add ASCII/Unicode board rendering
4. **Performance**: Optimize for speed if needed for training
5. **Card Expansion**: Add more cards to the pool
