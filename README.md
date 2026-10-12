# Qwen Choice

**简体中文** | [English](README.en.md)

基于 **Qwen3.5-4B** 的本地视觉判断工具。输入一张图片、一个问题和一组选项，通过一次模型前向计算，直接返回选项及其概率分布。

**单卡 RTX A6000 实测：完整请求中位数 169 ms，P95 为 174 ms。** 使用原始 BF16 权重，关闭思考，无训练或量化。
**切换其余Qwen模型直接可以按这个逻辑加速。**
## 示例

只回答一个问题：**图片中的客户消息属于哪类客服问题？**

选项为 `billing`（账单）、`technical`（技术）、`sales`（销售）和 `other`（其他）。

![虚构客服消息](examples/support_ticket.png)

图片中的客户反映重复扣款，模型返回 `billing`。消息和图片均为虚构示例，问题与选项定义见 [support_ticket.json](examples/support_ticket.json)。

## 性能

以下为启用视觉与语言模型编译后的测量结果，日期为 2026-09-21。

| 指标 | 结果 |
|---|---:|
| 完整请求中位数 | **168.694 ms** |
| 完整请求 P95（95 分位） | **173.989 ms** |
| GPU 模型阶段中位数 | 137.378 ms |
| 正式测量次数 | 30 |
| 匹配示例预期答案 | 30 / 30 |

测量条件：单卡 RTX A6000、BF16、batch=1；一张 1280×720 图片、一个问题、四个选项。官方处理器默认设置得到 880 个视觉 token、989 个总输入 token。

- **完整请求**从内存中的 RGB 图片开始，到选项与概率可用；包含图片处理、分词、CPU→GPU 传输、模型计算和结果读取，使用 CUDA 同步边界计时。
- **GPU 模型阶段**在输入传输后、结果回传前用 CUDA 事件测量，包含视觉编码和语言模型，也可能包含主机提交算子间的等待。
- 模型加载、图片文件读取、编译、预热、独立输入校验和日志写入不计入稳定请求耗时。固定问题的文字模板提前准备；图片及模型输出每次重新计算。
- 测量前预热 5 次。编译准备与这 5 次预热合计约 14.1 秒，使用机器已有编译缓存；其他环境的启动时间可能不同。

基准脚本交替运行两种输入准备路径，各测 30 次；上表取结果中的 `profiles.fast`。原始证据：[统计与校验](results/language_compile_v3/summary.json)、[逐次记录](results/language_compile_v3/records.jsonl)、[GPU 算子核查](results/operator_audit.json)。

这是一个固定示例的延迟测量，30 次重复不构成通用准确率评测。返回概率是对给定选项分数做 softmax 归一化的结果，未经概率校准。编译可能造成细微数值变化，不保证逐位等价。

## 工作方式

```text
图片 + 问题 + 选项
        ↓
官方图片处理器与分词器
        ↓
视觉编码器 + 语言模型：一次前向计算
        ↓
最后一个位置的输出分数
        ↓
选取答案字母对应分数 → softmax → 选项与概率
```

- **直接判断**：用单 token 字母表示选项，一次前向取得全部候选分数，跳过逐 token 生成。保留原始输出层，仅计算最后一个位置。
- **快速算子**：使用 causal-conv1d、FLA gated delta rule、融合 gated RMSNorm，以及 PyTorch SDPA 调用的 FlashAttention；运行记录确认对应 GPU 内核已执行。
- **模型编译**：视觉编码器使用 `torch.compile`；通过 `--compile-language` 同时编译语言模型。允许计算图切分，并记录实际执行情况。
- **减少准备开销**：复用固定问题的文字模板，把诊断哈希放到独立校验中；较大输入使用固定内存与非阻塞传输。

不跨请求复用图片、视觉特征、模型答案或语言状态。每个实例顺序处理请求。

## 安装

需要 NVIDIA CUDA GPU。实测环境为 Python 3.11、PyTorch 2.11.0+cu130、Transformers 5.13.1、FLA 0.5.2、Triton 3.6.0。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install packaging ninja
pip install --no-build-isolation -r requirements.txt
```

`causal-conv1d` 的构建需匹配 PyTorch、CUDA 和 GPU 架构。实测使用 `1.7.0+cu13torch2.11sm86` 的本地构建；缺少必要快速算子时，程序会报错，不会静默切换到慢速实现。

## 运行

使用 Hugging Face 模型：

```bash
python benchmark_fast.py \
  --model Qwen/Qwen3.5-4B \
  --compile-language \
  --output results/local-run
```

使用已有本地权重：

```bash
python benchmark_fast.py \
  --model /path/to/Qwen3.5-4B \
  --local-files-only \
  --compile-language \
  --output results/local-run
```

输出目录必须尚不存在。运行后查看 `summary.json` 中的 `profiles.fast`，逐次输出保存在 `records.jsonl`。修改图片尺寸、问题或选项后，应重新测量。

## 文件说明

| 文件 | 用途 |
|---|---|
| [qwen_choice.py](qwen_choice.py) | 单图单问推理、答案分数读取、快速算子验证 |
| [qwen_choice_fast.py](qwen_choice_fast.py) | 可复用的问题模板与输入准备 |
| [benchmark_fast.py](benchmark_fast.py) | 语言模型编译、配对计时与输入一致性校验 |
| [examples/](examples/) | 虚构消息图片、问题与选项 |
| [requirements.txt](requirements.txt) | Python 依赖 |

## 来源

问题设计借鉴 Jev 的 Choice 模式与公开客服分类案例，详见 [SOURCES.md](SOURCES.md)。本项目在本地运行 Qwen，不调用 Jev API，不需要 TypeSafe 密钥，也不隶属于 TypeSafe。

模型权重从 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 单独获取，其许可遵循模型发布方的条款。
