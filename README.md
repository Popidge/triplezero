# Triple Triad - MuZero Implementation

A reimplementation of the Triple Triad minigame from Final Fantasy 8, designed for use with MuZero self-play training.

## Overview

This project implements the basic "Open" ruleset of Triple Triad with a preselected set of 10 cards. The implementation follows the MuZero framework pattern, using the tictactoe example from the MuZero-general repository as a reference.

## Game Rules

### Basic Setup
- **Board**: 3×3 grid (9 positions)
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
- **North** = Top face
- **East** = Right face  
- **South** = Bottom face
- **West** = Left face

### Available Cards
| Card Name | North | East | South | West |
|-----------|-------|------|-------|------|
| Geezard | 1 | 4 | 5 | 1 |
| Fungar | 5 | 1 | 1 | 3 |
| Bite Bug | 1 | 3 | 3 | 5 |
| Red Bat | 6 | 1 | 1 | 2 |
| Blobra | 2 | 3 | 1 | 5 |
| Gayla | 2 | 1 | 4 | 4 |
| Gesper | 1 | 5 | 4 | 1 |
| Fastitocalon-F | 3 | 5 | 2 | 1 |
| Blood Soul | 2 | 1 | 6 | 1 |
| Caterchipillar | 4 | 2 | 4 | 3 |

## Project Structure

```
triplezero/
├── games/
│   ├── __init__.py
│   ├── abstract_game.py          # Base class for MuZero games
│   └── triple_triad/
│       ├── __init__.py
│       ├── cards.py              # Card definitions
│       ├── config.py             # MuZeroConfig for Triple Triad
│       ├── game.py               # Game wrapper (MuZero interface)
│       └── triple_triad.py       # Core game logic
├── tests/
│   ├── __init__.py
│   ├── test_triple_triad.py      # Core game logic tests
│   ├── test_game_wrapper.py      # Game wrapper tests
│   └── test_built_game.py        # Generated file validation
├── cards.txt                     # Card definitions (reference)
├── build_game.py                 # Build script with CI/CD pipeline
├── triple_triad.py               # Generated output for MuZero
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Install dependencies
uv sync

# Install development dependencies (for testing)
uv add --dev pytest pytest-cov
```

## Usage

### Basic Game Usage

```python
from games.triple_triad.game import Game

# Create a new game
game = Game(seed=42)

# Reset to initial state
observation = game.reset()

# Play a move (action is card_index * 9 + position)
action = 0  # Play first card at position 0
observation, reward, done = game.step(action)

# Get legal actions
legal_actions = game.legal_actions()

# Get current player
current_player = game.to_play()

# Render the game
game.render()
```

### Using the Core Game Logic

```python
from games.triple_triad.triple_triad import TripleTriad

# Create a new game
game = TripleTriad(seed=42)

# Reset to initial state
observation = game.reset()

# Get hands
hand_0 = game.get_hand(0)  # Player 0's cards
hand_1 = game.get_hand(1)  # Player 1's cards

# Get board state
board = game.get_board_state()

# Get winner
winner = game._get_winner()  # 0, 1, or None (draw)
```

### MuZero Training Configuration

```python
from games.triple_triad.config import MuZeroConfig

config = MuZeroConfig()

# Access configuration parameters
print(f"Observation shape: {config.observation_shape}")
print(f"Action space size: {len(config.action_space)}")
print(f"Max moves: {config.max_moves}")

# Get temperature for softmax
temperature = config.visit_softmax_temperature_fn(trained_steps=500)
```

## Action Space

The action space consists of 45 possible actions (5 cards × 9 positions):

```
action = card_index * 9 + board_position
```

- **card_index**: 0-4 (which card from hand to play)
- **board_position**: 0-8 (which board position to place it)

**Example**: Action 0 plays the first card at position 0 (top-left corner).

## Observation Space

The observation is a 3×3×11 NumPy array with the following channels:

| Channel | Description |
|---------|-------------|
| 0 | Player 0's cards (1 if owned by player 0, else 0) |
| 1 | Player 1's cards (1 if owned by player 1, else 0) |
| 2 | Current player indicator (1 for player 0, -1 for player 1) |
| 3 | North values for cards at each position |
| 4 | East values for cards at each position |
| 5 | South values for cards at each position |
| 6 | West values for cards at each position |

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=games --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_triple_triad.py -v
```

## Test Coverage

The project includes 94 comprehensive unit tests covering:

- **Card Definitions**: All 10 cards have valid values
- **Game Initialization**: Proper hand dealing and board setup
- **Legal Actions**: Correct action filtering based on hand and board state
- **Step Function**: Move execution, player switching, and state updates
- **Flipping Mechanics**: Card flipping when values are higher
- **Observations**: Correct encoding of game state
- **Winner Determination**: Win/loss/draw detection
- **Reward Calculation**: Proper reward values
- **Game Wrapper**: MuZero interface compliance
- **Configuration**: MuZeroConfig parameters
- **Generated File**: Build output validation

## Build Pipeline (CI/CD)

This project uses a modern development workflow with a modular codebase for development and a single-file output for MuZero integration:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development Phase                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  cards.py    │  │ triple_triad │  │   config.py  │          │
│  │  (card defs) │  │  (game logic)│  │  (MuZeroCfg) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                      │
│                  ┌──────────────┐                               │
│                  │   pytest     │  ← 94 tests                   │
│                  │  (validate)  │                               │
│                  └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼ (build_game.py)
┌─────────────────────────────────────────────────────────────────┐
│                    Build Phase                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               triple_triad.py                            │    │
│  │   (~28KB single file, MuZero-compatible)                │    │
│  │   - Card definitions                                    │    │
│  │   - AbstractGame base class                            │    │
│  │   - MuZeroConfig                                       │    │
│  │   - TripleTriad game logic                             │    │
│  │   - Game wrapper                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Deployment Phase                                 │
│          Drop into MuZero's games/ directory                    │
└─────────────────────────────────────────────────────────────────┘
```

### Build Commands

```bash
# Build the triple_triad.py file
python build_game.py

# Build and run all tests
python build_game.py --test

# Watch for changes and rebuild automatically
python build_game.py --watch

# Show help
python build_game.py --help
```

### Watch Mode

When running in watch mode (`python build_game.py --watch`), the script:
1. Monitors the `games/` directory for changes
2. Automatically rebuilds `triple_triad.py` when source files change
3. Runs the full test suite after each rebuild
4. Perfect for TDD (Test-Driven Development) workflows

### Generated File Structure

The generated `triple_triad.py` file contains all necessary components for MuZero:

```python
# Drop this file into MuZero's games/ directory
from triple_triad import Game, MuZeroConfig

game = Game(seed=42)
config = MuZeroConfig()
```

## Configuration Parameters

The MuZeroConfig class provides all hyperparameters needed for training:

### Game Configuration
- `observation_shape`: (3, 3, 11)
- `action_space`: 45 actions
- `max_moves`: 9
- `players`: [0, 1]

### Network Configuration
- `network`: "resnet"
- `blocks`: 1
- `channels`: 16
- `support_size`: 10

### Training Configuration
- `training_steps`: 1,000,000
- `batch_size`: 64
- `num_simulations`: 25
- `discount`: 1.0
- `lr_init`: 0.003

### Replay Buffer
- `replay_buffer_size`: 3000
- `num_unroll_steps`: 20
- `td_steps`: 20
- `PER`: True (Prioritized Experience Replay)

## Future Enhancements

Potential areas for expansion:

1. **Additional Rulesets**: Implement "Random", "Same", "Plus", "Combo" rules
2. **Expert Agent**: Develop a stronger heuristic agent for evaluation
3. **Visualization**: Add ASCII/Unicode board rendering
4. **Card Expansion**: Add more cards to the pool
5. **Performance Optimization**: Optimize for faster training

## Dependencies

- **Python**: ≥3.11
- **numpy**: Array operations
- **torch**: Neural network framework (for MuZero)
- **pytest**: Unit testing
- **pytest-cov**: Test coverage

## License

This project is for educational purposes, implementing a classic game mechanic for AI research.
