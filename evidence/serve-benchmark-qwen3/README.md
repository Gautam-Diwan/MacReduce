# Qwen3 placement study, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `mlx==0.32.4`, `mlx-lm==0.32.2`. Target `mlx-community/Qwen3-1.7B-4bit` at `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Draft `mlx-community/Qwen3-0.6B-4bit` at `73e3e38d981303bc594367cd910ea6eb48349da8`. Command:

```sh
.venv/bin/python -m macreduce bench --cycles 3 --max-tokens 32 --output evidence/serve-benchmark-qwen3
```

Three fresh-process cycles, 32 tokens, depths 0, 1, 2, 4, and 8. The Qwen2.5 study is unchanged in [serve-benchmark](../serve-benchmark/README.md).

**Measured:** 45 depth trials, no child errors, every trial hit the 32-token cap. Depths 0, 1, 2, and 4 matched greedy tokens on short, long, and code. Depth 8 matched long and code, and mismatched the short prompt at token 26 on all three replicates, so short depth-8 rates are absent.

Median decode, evaluation rows included in the three-sample median the scorer reports:

| Segment | Depth 0 tok/s | Fastest exact depth | Fastest tok/s | Fastest TTFT |
|---|---:|---:|---:|---:|
| short | 78.0 | 2 | 95.2 | depth 0, 0.039 s |
| long | 77.3 | 2 | 99.4 | depth 0, 1.494 s |
| code | 86.6 | 2 | 94.8 | depth 0, 0.124 s |

The cost model chose depth 1 for short, depth 2 for long, and depth 0 for code. It did not beat the hindsight depth on both time-to-first-token and decode rate in any segment. Long is a decode tie at depth 2 and does not improve time-to-first-token. The joint hypothesis is not met on this pair either.

One CPU-draft run on the short prompt matched tokens, 91.5 tok/s, time-to-first-token 0.101 s. That is one stream, not a concurrency claim. The long prefix was 1313 tokens. Prefill took 1.44 s. Reloading the cache took 1.73 ms and matched the recomputed continuation. The implied disk rate is a warm page cache of a file just written, 176,168,887 bytes. No second Mac was available. The ANE gate inside this report still does not convert the draft. `macreduce model-check` on the short prompt is in `model-check.json`: depth 2 matched greedy tokens, 17 of 32 tokens came from the draft.
