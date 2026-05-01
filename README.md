# Proactive Sound Effect Benchmark

This benchmark evaluates whether an audio–language model should **proactively speak** after hearing an **audio-only** cue (no transcript). Samples are split into **RESPOND** (user-relevant / safety / assistance scenarios) and **IGNORE** (ambient sounds where silence is appropriate). This repository ships **offline evaluation scripts** and **JSONL manifests**; audio files are hosted separately on Hugging Face.

| Resource | Link |
|----------|------|
| Code & manifests | [github.com/masaz14/Proactive-Sound-Effect-Benchmark](https://github.com/masaz14/Proactive-Sound-Effect-Benchmark) |
| Audio (Dataset) | **[masaz14/Proactive-Sound-Effect-Benchmark](https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark)** |

The official audio dataset is published as **`masaz14/Proactive-Sound-Effect-Benchmark`** at <https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark> (it matches the `./proactive-sound-effect/...` directory layout referenced by the GitHub manifests).

Download example (login may be required for gated/private assets):

```bash
pip install -U huggingface_hub
hf download masaz14/Proactive-Sound-Effect-Benchmark --repo-type=dataset --local-dir ./Proactive-Sound-Effect-Benchmark-data
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

You should run your model separately, write a predictions JSONL, then use this repository for offline alignment and scoring.

---

## Audio paths and layout

The manifest `path` fields are **relative paths** (the `./` prefix is recommended). The top-level audio directory is **`proactive-sound-effect`**, consistent with the Hugging Face layout. Examples:

```text
./proactive-sound-effect/Daily Living Sounds/Daily Affairs/IGNORE/<id>.wav
./proactive-sound-effect/Daily Living Sounds/Daily Affairs/RESPOND/<id>.wav
```

After downloading the dataset, place **`proactive-sound-effect`** and the two JSONL files in the same directory (or define your own root and join to absolute paths in your loader), so that each `path` resolves to a real local audio file.

---

## Expected model output format

The evaluator parses decisions and reply bodies from the raw model output (see `core.py`). The output must include these tags (case-insensitive):

```text
<Decision>RESPOND</Decision>   <!-- or IGNORE -->
<Reply>...</Reply>
```

Each row in the predictions JSONL should contain at least:

- `id`：与 manifest 一致  
- `_raw_reply` (recommended) or `reply`: the full raw model output string (the evaluator parses tags from it)

---

## Dependencies

- Python ≥ 3.10  
- For basic evaluation (no semantic matching), the standard library is sufficient to run `evaluate.py`.

Optional semantic matching (`standard_answers`):

```bash
pip install torch FlagEmbedding
```

Prepare a local reranker checkpoint directory (e.g. [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)) and pass it via `--reranker-dir`.

---

## Running evaluation

Run from a directory that contains `core.py`, `semantic.py`, and `evaluate.py`. If `evaluate.py` uses relative imports like `from .core import ...`, you must run it as a package; for a flat repo, consider switching to same-directory imports like `from core import ...` so you can run `python evaluate.py` directly.

**Decision accuracy only (no semantic matching):**

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/your_predictions.jsonl \
  --out /path/to/stats.json \
  --checkpoint-label my-model
```

**Enable semantic matching (compare RESPOND replies against reference answers):**

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/your_predictions.jsonl \
  --out /path/to/stats.json \
  --respond-jsonl proactive_reply_benchmark_response.jsonl \
  --reranker-dir /path/to/bge-reranker-v2-m3 \
  --semantic-threshold 0.5
```

The CLI prints the absolute path of the stats JSON and the overall accuracy **before/after semantic matching**. Per-domain/per-folder breakdowns are written to the `--out` JSON file.

---

## Metrics overview

- **Decision**: whether the predicted `RESPOND` / `IGNORE` matches the manifest `decision` (parsed from `<Decision>...</Decision>`).  
- **Semantic matching (optional)**: for ground-truth `RESPOND` items, score the `<Reply>` text against `standard_answers` using a reranker; if the score exceeds the threshold, it is counted as semantically correct. See `semantic.py` and the output stats JSON for aggregation details.

---
## Prompt examples
You are a proactively helpful model.
Your task is to determine whether to assist the user based on the provided audio.
- If the audio indicates physiological danger or environmental safety risks, provide appropriate assistance.
- If it reflects negative emotions, offer mental comfort.
- If it suggests equipment malfunction, provide warnings or safety guidance.
- If none of the above apply,just respond with "No Reply".Do not explain anything.
Your final output must strictly follow this format:
<Decision>(RESPOND or IGNORE)</Decision>
<Reply>(Your reply)</Reply>
## Citation



