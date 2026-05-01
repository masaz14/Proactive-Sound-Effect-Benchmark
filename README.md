# Proactive-Sound-Effect-Benchmark

English: This benchmark evaluates whether an audio–language model should **proactively speak** after hearing an **audio-only** cue (no transcript). Samples are split into **RESPOND** (user-relevant / safety / assistance scenarios) and **IGNORE** (ambient sounds where silence is appropriate). This repository ships **offline evaluation scripts** and **JSONL manifests**; audio files are hosted separately on Hugging Face.

中文：本基准评测模型在**仅有音效、无文本**输入时，是否能正确判断**是否需要主动回复**。样本分为需回复（**RESPOND**）与无需回复（**IGNORE**）。本 GitHub 仓库提供**离线评测代码**与清单文件；**音频数据请从 Hugging Face 下载**。

| Resource | Link |
|----------|------|
| Code & manifests | [github.com/masaz14/Proactive-Sound-Effect-Benchmark](https://github.com/masaz14/Proactive-Sound-Effect-Benchmark) |
| Audio (Dataset) | **[masaz14/Proactive-Sound-Effect-Benchmark](https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark)** |

官方音频数据集散名为 **`masaz14/Proactive-Sound-Effect-Benchmark`**，主页：<https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark>（与 GitHub 中 manifest 的 `./proactive-sound-effect/...` 目录结构对应）。

下载示例（需已登录：`huggingface-cli login`）：

```bash
pip install -U huggingface_hub
hf download masaz14/Proactive-Sound-Effect-Benchmark --repo-type=dataset --local-dir ./Proactive-Sound-Effect-Benchmark-data
```

或使用 `datasets` 库按行读取（具体列名以数据集实际 schema 为准）：

```python
from datasets import load_dataset

ds = load_dataset("masaz14/Proactive-Sound-Effect-Benchmark", split="train")
# 将音频保存到本地路径后，与 manifest 中的 path 对齐即可用于推理 / 评测
```

---

## Repository layout

| File | Role |
|------|------|
| `proactive_reply_benchmark.jsonl` | Full benchmark manifest: `id`, `path`, `description`, `decision` (`RESPOND` / `IGNORE`). |
| `proactive_reply_benchmark_response.jsonl` | Reference replies only for **RESPOND** items: `standard_answers` (list) per `id`, used for optional semantic matching. |
| `core.py` | Parses `<Decision>` / `<Reply>`, validity checks, path-based domain/folder grouping. |
| `semantic.py` | Optional reranker-based similarity vs `standard_answers`. |
| `evaluate.py` | CLI: align predictions to manifest, compute metrics, write stats JSON. |

模型推理与流式评测流水线不在本仓库；你需要自行跑模型并写出预测 JSONL，再用本仓库做离线对齐与打分。

---

## Audio paths and layout

Manifest 中的 `path` 为**相对路径**（建议带 `./` 前缀），且顶层目录名为 **`proactive-sound-effect`**，与 Hugging Face 上的目录一致，例如：

```text
./proactive-sound-effect/Daily Living Sounds/Daily Affairs/IGNORE/<id>.wav
./proactive-sound-effect/Daily Living Sounds/Daily Affairs/RESPOND/<id>.wav
```

下载数据集后，请将 **`proactive-sound-effect`** 与两份 JSONL 放在同一目录（或自行约定根目录并在加载音频时拼绝对路径），保证 `path` 能解析到真实文件。

---

## Expected model output format

评测从模型原始输出中解析决策与回复正文（见 `core.py`）。模型输出需包含如下标签（大小写不敏感）：

```text
<Decision>RESPOND</Decision>   <!-- or IGNORE -->
<Reply>...</Reply>
```

预测 JSONL 中每条样本至少包含：

- `id`：与 manifest 一致  
- `_raw_reply`（推荐）或 `reply`：模型完整原始字符串（含上述标签时，`evaluate` 按标签解析）

---

## Dependencies

- Python ≥ 3.10  
- 基础评测（不含语义匹配）：标准库即可运行 `evaluate.py`（建议安装本项目常用环境以便后续扩展）。

可选语义匹配（`standard_answers` + reranker）：

```bash
pip install torch FlagEmbedding
```

并准备本地 reranker 权重目录（如 [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 等），通过 `--reranker-dir` 传入。

---

## Running evaluation

在包含 `core.py`、`semantic.py`、`evaluate.py` 的目录下执行（若 `evaluate.py` 使用 `from .core import ...` 形式，需作为包运行或改为 `from core import ...`；**扁平仓库推荐改为同目录导入**，以便直接 `python evaluate.py`。）

**仅决策准确率（不使用语义匹配）：**

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/your_predictions.jsonl \
  --out /path/to/stats.json \
  --checkpoint-label my-model
```

**启用语义匹配（RESPOND 回复与参考答案列表比对）：**

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/your_predictions.jsonl \
  --out /path/to/stats.json \
  --respond-jsonl proactive_reply_benchmark_response.jsonl \
  --reranker-dir /path/to/bge-reranker-v2-m3 \
  --semantic-threshold 0.5
```

终端会打印 stats JSON 的绝对路径，以及 **semantic 前后**的整体准确率；按 domain / folder 的明细仅写入 `--out` 的 JSON 文件。

---

## Metrics overview

- **决策**：模型给出的 `RESPOND` / `IGNORE` 是否与 manifest 中的 `decision` 一致（基于 `<Decision>...</Decision>`）。  
- **语义匹配（可选）**：对 ground-truth 为 `RESPOND` 的样本，用 reranker 将 `<Reply>` 内文本与 `standard_answers` 比对，得分超过阈值则视为该维度通过；具体聚合方式见 `semantic.py` 与输出的 stats JSON。

---

## Citation



