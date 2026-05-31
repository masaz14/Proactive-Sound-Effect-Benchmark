# Proactive Sound Effect Benchmark

> Evaluates whether an audio–language model should **proactively speak** from **audio-only** cues (no transcript)—and, when it should, whether the reply is useful.

---

## What We Measure

| Capability | Description |
|------------|-------------|
| **Decision boundary** | Distinguish when to assist (`RESPOND`) vs stay silent (`IGNORE`) |
| **Reply quality** | For `RESPOND` items, whether the reply is relevant and helpful (optional semantic match) |

Each sample has a ground-truth `decision`. The `RESPOND` subset includes reference replies for optional semantic scoring.

---

## Resources

| Resource | Link |
|----------|------|
| Code & manifests | [github.com/masaz14/Proactive-Sound-Effect-Benchmark](https://github.com/masaz14/Proactive-Sound-Effect-Benchmark) |
| Audio (dataset) | [masaz14/Proactive-Sound-Effect-Benchmark](https://huggingface.co/datasets/masaz14/Proactive-Sound-Effect-Benchmark) |

The Hugging Face layout matches manifest `path` fields (root folder: `proactive-sound-effect`).

**Download** (login may be required for gated assets):

```bash
pip install -U huggingface_hub
hf download masaz14/Proactive-Sound-Effect-Benchmark \
  --repo-type=dataset \
  --local-dir ./Proactive-Sound-Effect-Benchmark-data
```

---

## Dataset Composition

**17** subcategories across **6** domains—daily living, human states, traffic, environment, music, and equipment—focused on proactive assistance and safety.

<details>
<summary>Subcategory counts (click to expand)</summary>

| Subcategory | Count | Domain |
|-------------|------:|--------|
| Daily Affairs | 126 | Daily Living |
| House Equipment States | 97 | Equipment |
| Physiological States | 68 | Human |
| Environment Background | 58 | Environment |
| Industrial Tools & Instruments | 43 | Equipment |
| Emotion Expression | 40 | Human |
| Housekeeping | 36 | Daily Living |
| Body Movements | 29 | Human |
| Vehicle | 27 | Traffic |
| Personal Care | 19 | Daily Living |
| Collective Ambience | 19 | Human |
| Ecological & Biological Context | 17 | Environment |
| Meteorological Dynamics | 17 | Environment |
| Artistic | 15 | Music |
| Traffic | 15 | Traffic |
| Geological Events & Hazards | 11 | Environment |
| Large Traffic | 7 | Traffic |

</details>

**Labels**

- **`RESPOND`** — User-relevant, safety, or assistance scenarios → model should speak up  
- **`IGNORE`** — Ambient or background sounds → silence is appropriate  

---

## Repository Layout

| File | Role |
|------|------|
| `proactive_reply_benchmark.jsonl` | Full manifest: `id`, `path`, `description`, `decision` |
| `proactive_reply_benchmark_response.jsonl` | `RESPOND` only: `standard_answers` per `id` (optional semantic match) |
| `core.py` | Parses `<Decision>` / `<Reply>`, path grouping, validity checks |
| `semantic.py` | Optional reranker similarity vs `standard_answers` |
| `evaluate.py` | CLI: align predictions, compute metrics, write stats JSON |

**Recommended layout** (each manifest `path` resolves to a local file):

```text
.
├── proactive_reply_benchmark.jsonl
├── proactive_reply_benchmark_response.jsonl
├── core.py
├── semantic.py
├── evaluate.py
└── proactive-sound-effect/
    └── Daily Living Sounds/
        └── Daily Affairs/
            ├── RESPOND/<id>.wav
            └── IGNORE/<id>.wav
```

Paths are relative; a `./` prefix is recommended, e.g.:

`./proactive-sound-effect/Daily Living Sounds/Daily Affairs/RESPOND/<id>.wav`

---

## Evaluation Workflow

```mermaid
flowchart LR
  A[Download audio] --> B[Model inference]
  B --> C[predictions.jsonl]
  C --> D[evaluate.py]
  D --> E[stats.json]
```

1. Run your model on every manifest audio item  
2. Write `predictions.jsonl`  
3. Score offline with this repo’s scripts  

### Expected Model Output

Tags are case-insensitive:

```text
<Decision>RESPOND</Decision>
<Reply>...</Reply>
```

Each line in `predictions.jsonl` needs at least:

- `id` — must match the manifest  
- `_raw_reply` (recommended) or `reply` — full raw model output string  

### Running Evaluation

**Decision accuracy only** (stdlib sufficient):

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/predictions.jsonl \
  --out /path/to/stats.json \
  --checkpoint-label my-model
```

**With semantic matching** (`RESPOND` replies vs `standard_answers`):

```bash
pip install torch FlagEmbedding
```

```bash
python evaluate.py \
  --benchmark proactive_reply_benchmark.jsonl \
  --pred /path/to/predictions.jsonl \
  --out /path/to/stats.json \
  --respond-jsonl proactive_reply_benchmark_response.jsonl \
  --reranker-dir /path/to/bge-reranker-v2-m3 \
  --semantic-threshold 0.5
```

> If `evaluate.py` uses `from .core import ...`, run it as a package. For a flat repo, switch to `from core import ...` and run `python evaluate.py` directly.

---

## Metrics

| Metric | Meaning |
|--------|---------|
| **Decision** | Predicted `RESPOND` / `IGNORE` matches manifest `decision` |
| **Semantic** (optional) | For ground-truth `RESPOND`, reranker score of `<Reply>` vs `standard_answers` exceeds threshold |

The CLI prints the stats JSON path and overall accuracy (before/after semantic matching). Per-domain and per-folder breakdowns are written to the `--out` JSON.

---

## Prompt Example

```text
You are a proactively helpful model.
Your task is to determine whether to assist the user based on the provided audio.
- If the audio indicates physiological danger or environmental safety risks, provide appropriate assistance.
- If it reflects negative emotions, offer mental comfort.
- If it suggests equipment malfunction, provide warnings or safety guidance.
- If none of the above apply, just respond with "No Reply". Do not explain anything.

Your final output must strictly follow this format:
<Decision>(RESPOND or IGNORE)</Decision>
<Reply>(Your reply)</Reply>
```

---

## Dependencies

- Python ≥ 3.10  
- Basic evaluation: standard library only  
- Semantic matching: `torch`, `FlagEmbedding`, and a local reranker (e.g. [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3))

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{proactive-sound-effect-benchmark,
  title        = {Proactive Sound Effect Benchmark},
  author       = {masaz14},
  year         = {2025},
  howpublished = {\url{https://github.com/masaz14/Proactive-Sound-Effect-Benchmark}}
}
```

Replace with your official publication entry when available.

---

## License

See the `LICENSE` file in the repository.
