"""Classic snake rules on a square grid; x grows right, y grows down."""
from __future__ import annotations

import random

DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


class SnakeGame:
    def __init__(self, size=20, *, seed=None, start_length=3):
        if size < 6:
            raise ValueError("Grid size must be at least 6")
        self.size = size
        self.start_length = start_length
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid = self.size // 2
        self.snake = [(mid - i, mid) for i in range(self.start_length)]
        self.direction = "right"
        self.score = 0
        self.steps = 0
        self.alive = True
        self.death_reason = None
        self.food = self._place_food()

    @property
    def head(self):
        return self.snake[0]

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def legal_directions(self):
        """Every direction except the 180-degree reversal, in fixed order."""
        return [d for d in DIRECTIONS if d != OPPOSITE[self.direction]]

    def next_head(self, direction):
        dx, dy = DIRECTIONS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def collision(self, direction):
        """Return 'wall', 'self', or None for moving one cell in direction."""
        cell = self.next_head(direction)
        if not self.in_bounds(cell):
            return "wall"
        eats = cell == self.food
        # The tail cell is vacated this step unless the snake grows.
        body = self.snake if eats else self.snake[:-1]
        return "self" if cell in body else None

    def step(self, direction):
        if not self.alive:
            raise RuntimeError("Game is over; call reset() first")
        if direction not in self.legal_directions():
            raise ValueError(f"Illegal direction {direction!r} while moving {self.direction}")
        self.direction = direction
        self.steps += 1
        crash = self.collision(direction)
        if crash:
            self.alive = False
            self.death_reason = crash
            return {"ate": False, "died": True, "reason": crash}
        cell = self.next_head(direction)
        ate = cell == self.food
        self.snake.insert(0, cell)
        if ate:
            self.score += 1
            self.food = self._place_food()
            if self.food is None:
                self.alive = False
                self.death_reason = "board_full"
                return {"ate": True, "died": True, "reason": "board_full"}
        else:
            self.snake.pop()
        return {"ate": ate, "died": False, "reason": None}

    def end(self, reason):
        """End the game without moving, e.g. when every legal move is deadly."""
        self.alive = False
        self.death_reason = reason
        return {"ate": False, "died": True, "reason": reason}

    def _place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    def snapshot(self):
        return {
            "grid": self.size,
            "snake": [list(c) for c in self.snake],
            "food": list(self.food) if self.food else None,
            "direction": self.direction,
            "score": self.score,
            "length": len(self.snake),
            "steps": self.steps,
            "alive": self.alive,
            "death_reason": self.death_reason,
        }
