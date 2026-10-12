# Jev-like Snake Playground

**简体中文** | [English](README.en.md)

本地贪吃蛇。每一步由代码算出合法走法的事实，再交给 **Qwen3.5** 在封闭选项里做一次判断，选出下一步。浏览器页面同步棋盘、选项概率和决策日志。

做法对齐 [Jev 的 Choice](https://docs.typesafe.ai/primitives/choice)：问题、选项和事实由程序给出，模型只在给定选项中选择。本仓库在本地运行 Qwen，不调用 Jev API，不需要 TypeSafe 密钥，也不隶属于 TypeSafe。详见 [SOURCES.md](SOURCES.md)。

## 运行

需要 NVIDIA CUDA GPU。权重另行准备，默认识别本地目录 `model/Qwen3.5-0.8B`。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install packaging ninja
pip install --no-build-isolation -r requirements.txt

python run.py --local-files-only
```

然后打开 `http://127.0.0.1:8765`。页面可以暂停，并调节两步之间的最短间隔。`Ctrl+C` 结束时，已完成的对局写入 `--output` 目录下的 `summary.json`，逐步记录在 `records.jsonl`。

换成本地其他 Qwen3.5 权重：

```bash
python run.py --model /path/to/Qwen3.5-2B --local-files-only
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--model` | `model/Qwen3.5-0.8B` | Hugging Face 名称或本地权重目录 |
| `--local-files-only` | 关闭 | 只读本地文件，不访问 Hub |
| `--host` / `--port` | `127.0.0.1` / `8765` | 页面地址 |
| `--grid` | `20` | 棋盘边长，至少为 6 |
| `--step-ms` | `150` | 两步之间的最短间隔（毫秒） |
| `--seed` | 随机 | 食物位置的随机种子 |
| `--max-games` | 一直运行 | 完成指定局数后退出 |
| `--output` | `results/playground-fast` | 日志目录，已存在时继续写入 |
| `--profile` | `eager` | `accelerated` 会按提示长度反复编译，对局会卡住 |

启动后先用 [examples/support_ticket.json](examples/support_ticket.json) 预热 5 次，再开始第一局。

## 一步怎么决定

```text
当前棋盘
    ↓
去掉 180° 回头；算出每个方向的事实
    ↓
致命走法不进入选项
    ↓
安全走法 ≥ 2：一次前向，对选项字母打分并 softmax
安全走法 = 1：直接走这一步
安全走法 = 0：判为死局（trapped）
    ↓
执行、刷新页面；死亡后停 1.5 秒并开新局
```

每个方向的事实包括：落点、是否撞墙或撞到自己、是否吃到食物、食物曼哈顿距离变化、该步之后还能到达的空格数。空格数少于蛇身长度时标为 `TRAP`。提示要求优先选择靠近食物的 `SAFE` 走法，并避开 `TRAP`。

模型侧不做逐 token 生成。选项名映射成单个大写字母，语言模型只跑一次，原始输出层只看最后一个位置，再在这些字母的分数上做 softmax。思考关闭。同一实例按顺序处理请求，不跨步复用答案。问题文本计划在题目和选项不变时复用，模型输出每次重算。

少于两个安全选项时不调用模型。

## 规则

- 正方形网格，x 向右、y 向下，蛇从中心、长度 3、朝右出发。
- 不能 180° 回头。撞墙或撞到自己则死亡。
- 吃到食物后长度和分数加 1，食物随机出现在空格上。
- 棋盘被占满时记为 `board_full`。

## 安装说明

实测环境为 Python 3.11、PyTorch 2.11.0+cu130、Transformers 5.13.1、FLA 0.5.2、Triton 3.6.0。`causal-conv1d` 必须与 PyTorch、CUDA 和 GPU 架构匹配。缺少这些快速算子时程序会报错，不会静默退回慢速实现。

## 文件

| 路径 | 作用 |
|---|---|
| [run.py](run.py) | 加载模型、驱动对局、写日志 |
| [qwen_choice.py](qwen_choice.py) | 封闭选项的单次前向与分数读取 |
| [qwen_choice_fast.py](qwen_choice_fast.py) | 复用不变的问题文本计划 |
| [playground/snake.py](playground/snake.py) | 贪吃蛇规则 |
| [playground/prompt.py](playground/prompt.py) | 把棋盘变成问题和选项事实 |
| [playground/server.py](playground/server.py) | 轮询用的本地页面服务 |
| [playground/static/index.html](playground/static/index.html) | 棋盘与决策日志 |
| [requirements.txt](requirements.txt) | Python 依赖 |

模型权重从 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 或其他 Qwen3.5 因果语言模型单独获取，许可遵循模型发布方的条款。

## 致谢

封闭选项的单次前向判断来自 [RJMSWD/QwenJev](https://github.com/RJMSWD/QwenJev)。该项目把 Qwen3.5 改造成 Jev 式的本地选择，并给出视觉判断的加速路径。本仓库在此基础上做贪吃蛇对局。
