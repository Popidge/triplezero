#!/usr/bin/env python3
"""
Build script to generate a single flat triple_triad.py file for MuZero.

This script concatenates all the modular game files into a single file
that can be dropped into MuZero's games/ directory.

Usage:
    python build_game.py          # Build the file
    python build_game.py --test   # Build and run tests
    python build_game.py --watch  # Watch for changes and rebuild
"""
import importlib.util
import pathlib
import subprocess
import sys
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def get_file_content(file_path: pathlib.Path, strip_docstring: bool = True) -> str:
    """Read file content, optionally stripping the module docstring."""
    content = file_path.read_text()
    if strip_docstring:
        # Strip the module docstring (first string literal)
        if content.startswith('"""') or content.startswith("'''"):
            end_marker = content.find('"""') if content.startswith('"""') else content.find("'''")
            if end_marker != -1:
                # Find the end marker after the start
                end_idx = content.find('"""', 3) if content.startswith('"""') else content.find("'''", 3)
                if end_idx != -1:
                    content = content[end_idx + 3:].strip()
                    # Remove leading blank lines
                    content = content.lstrip('\n')
    return content


def build_triple_triad():
    """Build the single triple_triad.py file."""
    project_root = pathlib.Path(__file__).parent
    
    # Read the card definitions
    cards_file = project_root / "games" / "triple_triad" / "cards.py"
    abstract_game_file = project_root / "games" / "abstract_game.py"
    triple_triad_file = project_root / "games" / "triple_triad" / "triple_triad.py"
    config_file = project_root / "games" / "triple_triad" / "config.py"
    game_file = project_root / "games" / "triple_triad" / "game.py"
    
    # Verify files exist
    for f in [cards_file, abstract_game_file, triple_triad_file, config_file, game_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Get content from each file
    cards_content = get_file_content(cards_file)
    abstract_game_content = get_file_content(abstract_game_file)
    triple_triad_content = get_file_content(triple_triad_file)
    config_content = get_file_content(config_file)
    game_content = get_file_content(game_file)
    
    # Remove the cards import from triple_triad since we're inlining it
    triple_triad_content = triple_triad_content.replace(
        "from .cards import TRIPLE_TRIAD_CARDS\n\n",
        ""
    )
    
    # Strip imports from game_content that won't exist when dropped into MuZero
    # The Game class will use the locally-defined AbstractGame and TripleTriad
    game_content = game_content.replace(
        "from games.abstract_game import AbstractGame\n",
        ""
    )
    game_content = game_content.replace(
        "from games.triple_triad.triple_triad import TripleTriad\n",
        ""
    )
    
    # Build the content
    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append(f'# Auto-generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('# Generated from modular Triple Triad implementation')
    lines.append('# Drop this file into MuZero\'s games/ directory')
    lines.append('')
    lines.append('import datetime')
    lines.append('import pathlib')
    lines.append('')
    lines.append('import numpy as np')
    lines.append('import torch')
    lines.append('')
    lines.append('__version__ = "1.0.0"')
    lines.append('')
    lines.append('')
    lines.append('# ============== CARD DEFINITIONS ==============')
    lines.append(cards_content)
    lines.append('')
    lines.append('')
    lines.append('# ============== ABSTRACT GAME BASE ==============')
    lines.append(abstract_game_content)
    lines.append('')
    lines.append('')
    lines.append('# ============== MUZERO CONFIG ==============')
    lines.append(config_content)
    lines.append('')
    lines.append('')
    lines.append('# ============== CORE GAME LOGIC ==============')
    lines.append(triple_triad_content)
    lines.append('')
    lines.append('')
    lines.append('# ============== GAME WRAPPER ==============')
    lines.append(game_content)
    lines.append('')
    
    # Write the output file
    output_file = project_root / "triple_triad.py"
    output_file.write_text('\n'.join(lines))
    
    print(f"✓ Generated: {output_file}")
    print(f"  Size: {len(output_file.read_text())} bytes")
    
    return output_file


def run_tests():
    """Run the test suite."""
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=pathlib.Path(__file__).parent
    )
    
    if result.returncode == 0:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return result.returncode


def smoke_test_generated_file(output_file: pathlib.Path) -> bool:
    """Verify the generated file can be imported and instantiated in isolation.
    
    This simulates dropping the file into MuZero's games/ directory by
    loading it as a standalone module without any project imports.
    
    Args:
        output_file: Path to the generated triple_triad.py file.
        
    Returns:
        True if the file passes all smoke tests, False otherwise.
    """
    print("\n" + "=" * 60)
    print("SMOKE TEST - Verifying generated file in isolation")
    print("=" * 60)
    
    if not output_file.exists():
        print(f"✗ Generated file not found: {output_file}")
        return False
    
    try:
        # Load the module as if it were dropped into MuZero's games/ directory
        spec = importlib.util.spec_from_file_location("triple_triad_generated", output_file)
        module = importlib.util.module_from_spec(spec)
        
        # This will fail if there are broken imports
        spec.loader.exec_module(module)
        print("✓ File imports successfully")
        
        # Test instantiation
        game = module.Game(seed=42)
        print("✓ Game instantiates successfully")
        
        # Test basic interface
        obs = game.reset()
        print(f"✓ Reset returns observation with shape: {obs.shape}")
        
        # Verify observation shape matches expected (3, 3, 11)
        expected_shape = (3, 3, 11)
        if obs.shape == expected_shape:
            print(f"✓ Observation shape matches expected: {expected_shape}")
        else:
            print(f"✗ Observation shape mismatch! Got {obs.shape}, expected {expected_shape}")
            return False
        
        # Test legal_actions returns expected range
        legal = game.legal_actions()
        print(f"✓ Legal actions count: {len(legal)}")
        if 0 <= len(legal) <= 45:
            print(f"✓ Legal actions in expected range [0, 45]")
        else:
            print(f"✗ Legal actions count out of range: {len(legal)}")
            return False
        
        # Test MuZeroConfig
        config = module.MuZeroConfig()
        print("✓ MuZeroConfig instantiates successfully")
        
        # Verify action_space length
        expected_action_count = 45  # 5 cards * 9 positions
        if len(config.action_space) == expected_action_count:
            print(f"✓ Action space has expected {expected_action_count} actions")
        else:
            print(f"✗ Action space has {len(config.action_space)} actions, expected {expected_action_count}")
            return False
        
        print("\n✓ All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class RebuildHandler(FileSystemEventHandler):
    """Handler for file changes that triggers rebuild."""
    
    def __init__(self, project_root: pathlib.Path):
        self.project_root = project_root
        self.debounce_time = 1.0
        self.last_rebuild = 0
    
    def on_any_event(self, event):
        current_time = time.time()
        if current_time - self.last_rebuild < self.debounce_time:
            return
        
        if event.src_path.endswith(('.py', '.txt')):
            # Only rebuild if it's one of our source files
            rel_path = pathlib.Path(event.src_path).relative_to(self.project_root)
            if str(rel_path).startswith('games/') or str(rel_path) == 'cards.txt':
                self.last_rebuild = current_time
                print(f"\n{'=' * 60}")
                print(f"Detected change: {event.src_path}")
                print(f"Rebuilding...")
                print("=" * 60)
                try:
                    build_triple_triad()
                    print("\nRebuild complete. Running tests...")
                    run_tests()
                except Exception as e:
                    print(f"Error during rebuild: {e}")


def watch_mode():
    """Watch for file changes and rebuild automatically."""
    project_root = pathlib.Path(__file__).parent
    
    print("=" * 60)
    print("WATCH MODE - Press Ctrl+C to stop")
    print("=" * 60)
    print(f"Watching: {project_root / 'games'}")
    print()
    
    event_handler = RebuildHandler(project_root)
    observer = Observer()
    observer.schedule(event_handler, str(project_root / 'games'), recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print(__doc__)
            return
        
        if sys.argv[1] == '--test':
            print("=" * 60)
            print("BUILDING AND TESTING")
            print("=" * 60)
            output_file = build_triple_triad()
            print()
            smoke_passed = smoke_test_generated_file(output_file)
            print()
            test_passed = run_tests() == 0
            sys.exit(0 if (smoke_passed and test_passed) else 1)
        
        if sys.argv[1] == '--watch':
            watch_mode()
            return
    
    print("=" * 60)
    print("BUILDING triple_triad.py")
    print("=" * 60)
    output_file = build_triple_triad()
    print()
    smoke_test_generated_file(output_file)
    print("\nTo build and test: python build_game.py --test")
    print("To watch for changes: python build_game.py --watch")


if __name__ == "__main__":
    main()
