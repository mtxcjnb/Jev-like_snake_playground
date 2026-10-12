"""Snake playground driven by one closed-set choice per move."""
from .prompt import build_choice
from .server import PlaygroundState, start_server
from .snake import DIRECTIONS, SnakeGame

__all__ = ["DIRECTIONS", "PlaygroundState", "SnakeGame", "build_choice", "start_server"]
