# 调用主函数
import argparse
import torch
from pathlib import Path
from qwen_choice_fast import FastQwenChoice
import json
import os
import time
from types import SimpleNamespace

from playground import PlaygroundState, SnakeGame, build_choice, start_server

ROOT = Path(__file__).resolve().parent
STATUS_PATH = ROOT / 'playground_choices/playground_status.json'
GAME_OVER_PAUSE_S = 1.5

PG = SimpleNamespace(game=None, state=None, server=None, current=None, step_started=0.0,
                     records=None, output=None, max_games=None)


def init_playground(args):
    """初始化游戏、共享状态与 Web 页面。"""
    PG.game = SnakeGame(args.grid, seed=args.seed)
    PG.state = PlaygroundState(step_ms=args.step_ms)
    PG.state.set_board(PG.game.snapshot(), phase='running')
    PG.server = start_server(PG.state, args.host, args.port)
    PG.output = args.output
    PG.records = (args.output / 'records.jsonl').open('w', encoding='utf-8')
    PG.max_games = args.max_games
    print(f'[playground] open http://{args.host}:{args.port}', flush=True)


def get_playground_status():
    """获取 playground 状态并写入 playground_status.json 文件。"""
    PG.state.wait_while_paused()
    PG.step_started = time.perf_counter()
    PG.current = build_choice(PG.game)
    # Atomic replace: readers (and editors with the file open) never see a partial write.
    tmp = STATUS_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(PG.current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, STATUS_PATH)


def forced_result(case):
    """少于 2 个非致命选项时不调用模型：唯一安全方向直接走，全部致命则判为死局。"""
    options = list(case['options'])
    choice = options[0] if options else None
    return {'choice': choice, 'probabilities': {choice: 1.0} if choice else {},
            'request_s': 0.0, 'model_gpu_s': 0.0, 'audit': {'input_tokens': 0},
            'forced': 'only_safe_move' if options else 'no_safe_move'}


def update_playground_status(result):
    """按模型选择执行一步，记录日志，死亡时开新局。"""
    choice = result['choice']
    if result.get('forced') == 'no_safe_move':
        step = PG.game.steps
        event = PG.game.end('trapped')
    else:
        if choice not in PG.current['options']:
            raise RuntimeError(f'Model returned {choice!r}, not one of {list(PG.current["options"])}')
        step = PG.game.steps + 1
        event = PG.game.step(choice)
    entry = PG.state.add_decision({
        'step': step,
        'question': PG.current['question'],
        'options': PG.current['options'],
        'facts': PG.current['facts'],
        'choice': choice,
        'probabilities': result.get('probabilities'),
        'request_s': result['request_s'],
        'model_gpu_s': result['model_gpu_s'],
        'input_tokens': result['audit']['input_tokens'],
        'forced': result.get('forced'),
        'event': event,
    })
    PG.records.write(json.dumps(entry, ensure_ascii=False) + '\n')
    PG.records.flush()

    if event['died']:
        snap = PG.game.snapshot()
        PG.state.set_board(snap, phase='game_over')
        PG.state.finish_game({'score': snap['score'], 'steps': snap['steps'],
                              'length': snap['length'], 'reason': snap['death_reason']})
        print(f"[playground] game over: score={snap['score']} steps={snap['steps']} "
              f"reason={snap['death_reason']}", flush=True)
        time.sleep(GAME_OVER_PAUSE_S)
        PG.game.reset()
        PG.state.set_board(PG.game.snapshot(), phase='running')
        return
    PG.state.set_board(PG.game.snapshot())
    remaining = PG.state.get_step_ms() / 1000 - (time.perf_counter() - PG.step_started)
    if remaining > 0:
        time.sleep(remaining)


def playground_running():
    return PG.max_games is None or len(PG.state.games) < PG.max_games


def close_playground():
    summary = PG.state.summary()
    (PG.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    PG.records.close()
    PG.server.shutdown()
    print(f"[playground] {summary['games_finished']} games, best={summary['best_score']}, "
          f"mean={summary['mean_score']:.2f}, mean request={summary['mean_request_ms']:.1f} ms",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='model/Qwen3.5-0.8B')
    ap.add_argument('--local-files-only', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('results/playground-fast'))
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--grid', type=int, default=20)
    ap.add_argument('--step-ms', type=int, default=150, help='minimum interval between moves')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--max-games', type=int, default=None, help='stop after N games (default: forever)')
    ap.add_argument('--profile', choices=['eager', 'accelerated'], default='eager',
                    help='accelerated uses static-shape torch.compile, which recompiles for '
                         'every new prompt length and stalls the game')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    model = FastQwenChoice(args.model, local_files_only=args.local_files_only)
    fast_prepare = FastQwenChoice.prepare.__get__(model)

    model.prepare = fast_prepare

    # 预热模型
    case = json.loads((root / 'examples/support_ticket.json').read_text())
    for _ in range(5):
        model.judge(case['question'], case['options'], profile=args.profile)

    # 初始化游戏
    init_playground(args)

    try:
        while playground_running():
            get_playground_status()
            case = json.loads((root / 'playground_choices/playground_status.json').read_text())
            if len(case['options']) >= 2:
                result = model.judge(case['question'], case['options'], profile=args.profile)
            else:
                result = forced_result(case)
            update_playground_status(result)
    except KeyboardInterrupt:
        pass
    finally:
        close_playground()


if __name__ == "__main__":
    main()
