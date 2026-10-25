# Qwen3 greedy equivalence, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `mlx==0.32.4`, `mlx-lm==0.32.2`. Commands:

```sh
.venv/bin/python -m macreduce equivalence --max-tokens 32 --depth 1 --output evidence/serve-equivalence-qwen3/depth-1.json
.venv/bin/python -m macreduce equivalence --max-tokens 32 --depth 2 --output evidence/serve-equivalence-qwen3/result.json
.venv/bin/python -m macreduce equivalence --max-tokens 32 --depth 4 --output evidence/serve-equivalence-qwen3/depth-4.json
.venv/bin/python -m macreduce equivalence --max-tokens 32 --depth 8 --output evidence/serve-equivalence-qwen3/depth-8.json
```

Target `mlx-community/Qwen3-1.7B-4bit` at `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Draft `mlx-community/Qwen3-0.6B-4bit` at `73e3e38d981303bc594367cd910ea6eb48349da8`. The loader passed those revisions. Both repos were last modified 2025-04-28.

The Qwen2.5 study is unchanged in [serve-equivalence](../serve-equivalence/README.md) and [serve-benchmark](../serve-benchmark/README.md). Gemma 4 E4B plus E2B 4-bit is about 8.8 GB and was not downloaded: this volume had 3.0 GiB free. Qwen3.5 was not pinned. In the installed `mlx-lm` 0.32.2, `qwen3_5.py` `make_cache` returns `ArraysCache` for linear-attention layers, and `ArraysCache.is_trimmable()` is false, so stock speculative decoding refuses that cache.

**Measured:** depths 1, 2, and 4 matched greedy tokens for 32 tokens on the short, long, and code prompts. Every one of those generations has length 32, so an early stop was not observed. Depth 8 matched long and code, and mismatched the short prompt at token index 26. This is not a timing result and not a depth sweep. The Qwen2.5 tok/s numbers do not transfer.
