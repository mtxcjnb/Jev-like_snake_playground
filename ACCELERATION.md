# 算子核查与进一步提速

首轮 0.206 秒是一个包含诊断开销的参考实现，不应称为速度极限或“所有优化均已启用”。

## 哪些加速实际运行了

| 项目 | 实际状态 |
|---|---|
| 直接读取候选分数，跳过逐 token 生成 | 已启用，一次完整前向 |
| causal-conv1d 快速卷积 | 已启用，24 个线性注意力层 |
| FLA gated delta rule | 已启用，24 个线性注意力层 |
| 融合 gated RMSNorm | 已启用，24 个线性注意力层 |
| FlashAttention | **已启用**，通过 PyTorch SDPA 调用；Profiler 看到 `pytorch_flash::flash_fwd_kernel` |
| 视觉编码器 torch.compile | 已启用，允许图切分，检查实际编译图执行 |
| 固定内存、非阻塞传输 | 已启用 |
| 固定问题的文字模板复用 | 新增；不复用图片、视觉特征或答案 |
| 语言模型 torch.compile | 新增可选开关 `--compile-language`；允许图切分 |
| 语言模型 CUDA Graph | 独立测试完成，但未观察到明显收益，未采用 |

具体算子名称与 CUDA Graph 对照保存在 [operator_audit.json](results/operator_audit.json)。

首轮没有安装独立 FlashAttention 软件包，不等于没有使用 FlashAttention 算法：本环境的 PyTorch 已提供对应 GPU 内核，运行记录确认它被调用。

## 新测量

继续使用同一张 1280×720 图片、880 个视觉 token、989 个总输入 token、单卡 RTX A6000、BF16、一个问题。每个配置测 30 次，预热单列。

| 轮次 | 配置 | 完整请求中位数 | P95 | GPU 模型阶段中位数 |
|---|---|---:|---:|---:|
| v2 配对 | 原图片校验路径，视觉编译 | 199.821 ms | 203.538 ms | 147.391 ms |
| v2 配对 | 文字模板复用，移除逐次诊断哈希 | **182.041 ms** | **184.178 ms** | **147.677 ms** |
| v3 配对 | 保留图片校验，视觉与语言均编译 | 187.166 ms | 191.878 ms | 137.023 ms |
| v3 配对 | 模板复用、移除诊断哈希、视觉与语言均编译 | **168.694 ms** | **173.989 ms** | **137.378 ms** |

每轮内部交替运行两种准备方式，复用同一个已加载模型。v2/v3 是两个独立进程的测量，语言编译收益不能当成同进程随机交替实验。首轮 205.757 ms 与 v2 的 199.821 ms 也存在运行波动，不把这部分差异归因于代码改进。

图片张量约 21.6 MB，单独测量其诊断 SHA256 的中位耗时约 **18.1 ms**。新路径把完整张量一致性验证放在计时前执行，正常请求仅保留必要的尺寸与类型信息，不重复生成这个诊断哈希。图片仍由官方处理器每次重新处理；没有降低分辨率、量化或复用模型输出。

v3 的语言编译实际生成 16 个子图，预热和测量期间累计执行 17,940 次子图。准备与 5 次预热合计约 14.1 秒，未计入稳定请求耗时；不代表清空机器编译缓存后的首次成本。

## 正确性边界

- 固定问题模板优化前后，实际处理器输出的所有张量逐个完全一致；额外使用白色空图再切回原图，检查图片输入不会被错误复用。
- v2 的 30 对结果，选项与概率全部相同。
- v3 同轮不同准备方式的概率全部相同。
- v2 与 v3 的最终选择均为 `billing`，但最大选项概率差约 **0.00016576**，原始候选分数也有变化。因此不能宣称语言编译数值完全等价。
- 所有参数的内存位置和版本计数均保持不变。验证仍只覆盖这个示例，不构成通用准确率评测。

原始记录：[v2](results/request_fast_v2/summary.json)、[v3](results/language_compile_v3/summary.json)、[编译前后数值对照](results/compile_comparison.json)。

CUDA Graph 独立试验中，旧路径约 198.4 ms，加入语言计算图后约 198.5 ms，选项概率相同。当前输入下没有值得保留的收益。CUDA Graph 的作用主要是减少主机下发算子的开销，不能省去矩阵乘法本身；参见 [PyTorch 2.11 文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)。

## 使用

概率保持与原配置一致的请求优化：

```bash
python benchmark_fast.py --model /path/to/Qwen3.5-4B --local-files-only \
  --output results/local-fast
```

更快的语言编译配置：

```bash
python benchmark_fast.py --model /path/to/Qwen3.5-4B --local-files-only \
  --compile-language --output results/local-language-compiled
```

正常代码可将 `QwenChoice` 换为 `qwen_choice_fast.FastQwenChoice` 使用模板优化。语言编译的启用与执行计数实现见 `benchmark_fast.py`。
