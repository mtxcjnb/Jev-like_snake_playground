"""Turn a snake position into one closed-set question with per-move facts.

Code computes the facts; the model only chooses among the legal moves.
"""
from __future__ import annotations

from collections import deque

from .snake import DIRECTIONS, SnakeGame


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _free_space(game: SnakeGame, head, body):
    """Cells reachable from head when body (including head) blocks movement."""
    blocked = set(body)
    seen = {head}
    queue = deque([head])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            cell = (x + dx, y + dy)
            if cell not in seen and cell not in blocked and game.in_bounds(cell):
                seen.add(cell)
                queue.append(cell)
    return len(seen) - 1


def move_facts(game: SnakeGame, direction):
    cell = game.next_head(direction)
    before = _manhattan(game.head, game.food)
    crash = game.collision(direction)
    if crash:
        return {"head": list(cell), "deadly": True, "reason": crash, "eats": False,
                "food_distance_before": before, "food_distance_after": None,
                "free_space": 0, "trap": False}
    eats = cell == game.food
    body = [cell] + (game.snake if eats else game.snake[:-1])
    space = _free_space(game, cell, body)
    return {"head": list(cell), "deadly": False, "reason": None, "eats": eats,
            "food_distance_before": before,
            "food_distance_after": 0 if eats else _manhattan(cell, game.food),
            "free_space": space, "trap": space < len(body)}


def describe(facts, length_after):
    x, y = facts["head"]
    if facts["trap"]:
        tag = (f"TRAP: only {facts['free_space']} free cells reachable, "
               f"fewer than snake length {length_after}.")
    else:
        tag = "SAFE."
    parts = [tag, f"Head moves to ({x},{y})."]
    if facts["eats"]:
        parts.append("EATS FOOD.")
    else:
        before, after = facts["food_distance_before"], facts["food_distance_after"]
        trend = "closer" if after < before else "farther"
        parts.append(f"Food distance {before} -> {after} ({trend}).")
    parts.append(f"Free space after move: {facts['free_space']} cells.")
    return " ".join(parts)


def build_choice(game: SnakeGame):
    if not game.alive or game.food is None:
        raise RuntimeError("No decision for a finished game")
    n = game.size
    hx, hy = game.head
    fx, fy = game.food
    question = (
        f"You control a snake on a {n}x{n} grid. x increases to the right, y increases "
        f"downward; valid cells are (0,0) to ({n - 1},{n - 1}).\n"
        "Rules: moving into a wall or the snake's own body ends the game. "
        "Eating food adds 1 to length and score.\n"
        f"Current state: head ({hx},{hy}), moving {game.direction}, length {len(game.snake)}, "
        f"food at ({fx},{fy}), score {game.score}.\n"
        "Choose the next move. Prefer a SAFE move that brings the head closer to the food; "
        "avoid moves marked TRAP."
    )
    # Deadly moves stay in facts for auditing but are never offered to the model.
    facts, options = {}, {}
    for direction in game.legal_directions():
        f = move_facts(game, direction)
        facts[direction] = f
        if not f["deadly"]:
            options[direction] = describe(f, len(game.snake) + int(f["eats"]))
    return {"question": question, "options": options, "facts": facts,
            "state": game.snapshot()}
