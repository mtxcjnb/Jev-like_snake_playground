# Jev-like Snake Playground

[简体中文](README.md) | **English**

A local game of snake. Each step, code computes facts for the legal moves, then **Qwen3.5** picks one option from that closed set. A browser page shows the board, option probabilities, and the decision log.

The setup follows [Jev's Choice](https://docs.typesafe.ai/primitives/choice): the program supplies the question, options, and facts, and the model only selects among those options. This repository runs Qwen locally. It does not call the Jev API, does not need a TypeSafe key, and is not affiliated with TypeSafe. See [SOURCES.md](SOURCES.md).

## Run

Requires an NVIDIA CUDA GPU. Weights are obtained separately. The default path is `model/Qwen3.5-0.8B`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install packaging ninja
pip install --no-build-isolation -r requirements.txt

python run.py --local-files-only
```

Open `http://127.0.0.1:8765`. The page can pause the game and change the minimum gap between moves. On `Ctrl+C`, finished games are written to `summary.json` under `--output`, and per-step records go to `records.jsonl`.

Point `--model` at other local Qwen3.5 weights:

```bash
python run.py --model /path/to/Qwen3.5-2B --local-files-only
```

| Flag | Default | Role |
|---|---|---|
| `--model` | `model/Qwen3.5-0.8B` | Hugging Face id or a local weight directory |
| `--local-files-only` | off | Read local files only; do not contact the Hub |
| `--host` / `--port` | `127.0.0.1` / `8765` | Page address |
| `--grid` | `20` | Board size; minimum 6 |
| `--step-ms` | `150` | Minimum milliseconds between moves |
| `--seed` | random | Seed for food placement |
| `--max-games` | run until stopped | Exit after this many finished games |
| `--output` | `results/playground-fast` | Log directory; an existing directory is reused |
| `--profile` | `eager` | `accelerated` recompiles for each new prompt length and stalls the game |

Startup warms the model with five passes over [examples/support_ticket.json](examples/support_ticket.json) before the first game.

## One decision

```text
Current board
    ↓
Drop the 180° reversal and compute facts for each direction
    ↓
Deadly moves are left out of the options
    ↓
2 or more safe moves: one forward pass, score the option letters, softmax
1 safe move: take it
0 safe moves: end the game as trapped
    ↓
Apply the move and refresh the page; after death, wait 1.5s and start a new game
```

Facts for a direction are the landing cell, wall or self collision, whether it eats, the change in Manhattan distance to the food, and how many empty cells remain reachable. A move is marked `TRAP` when that free space is smaller than the snake. The prompt prefers a `SAFE` move that gets closer to the food and tells the model to avoid `TRAP`.

The model does not generate token by token. Option names map to single uppercase letters. The language model runs once, the original output head is applied only at the last position, and those letter scores are softmax-normalized. Thinking is disabled. One instance handles requests in order and does not reuse answers across steps. The text plan is reused when the question and options are unchanged; model outputs are recomputed every call.

The model is not called when fewer than two safe moves remain.

## Rules

- Square grid, x to the right and y downward. The snake starts at the center with length 3, facing right.
- A 180° reversal is illegal. Hitting a wall or the snake's own body ends the game.
- Eating food adds 1 to length and score. The next food cell is chosen at random among empty cells.
- A completely filled board is recorded as `board_full`.

## Install notes

Tested with Python 3.11, PyTorch 2.11.0+cu130, Transformers 5.13.1, FLA 0.5.2, and Triton 3.6.0. The `causal-conv1d` build must match PyTorch, CUDA, and the GPU architecture. Missing fast kernels raise an error instead of falling back to a slow path.

## Files

| Path | Role |
|---|---|
| [run.py](run.py) | Load the model, drive games, write logs |
| [qwen_choice.py](qwen_choice.py) | One forward pass and score readout for a closed set |
| [qwen_choice_fast.py](qwen_choice_fast.py) | Reuse an unchanged question text plan |
| [playground/snake.py](playground/snake.py) | Snake rules |
| [playground/prompt.py](playground/prompt.py) | Turn a board into a question and option facts |
| [playground/server.py](playground/server.py) | Local page server with polling |
| [playground/static/index.html](playground/static/index.html) | Board and decision log |
| [requirements.txt](requirements.txt) | Python dependencies |

Weights come separately from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or another Qwen3.5 causal language model, under that publisher's license.
