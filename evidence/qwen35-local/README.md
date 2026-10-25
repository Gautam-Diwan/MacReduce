# Qwen3.5 speculative check, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `mlx==0.32.4`, `mlx-lm==0.32.2`.

Target `mlx-community/Qwen3.5-2B-4bit` at `674aaa7240b91e8012fcad5d791b7dfe5ba90207`. Draft `mlx-community/Qwen3.5-0.8B-4bit` at `da28692b5f139cb0ec58a356b437486b7dac7462`. Both tokenizers report vocab size 248044. The 2B cache is 18 `ArraysCache` layers and 6 `KVCache` layers. `ArraysCache.is_trimmable()` is false.

**Measured:** plain generation of the raw prompt `Reply with one word: ready` produced 32 tokens at 80.86 tok/s, including load. Speculative decoding at depth 2 raises `ValueError: Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'}).`

**4B check, config only:** `mlx-community/Qwen3.5-4B-4bit` at `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` is the same `qwen3_5` class. It has 32 layers and `full_attention_interval` 4, so 24 layers are linear attention and 8 are full attention. `make_cache` in the installed `qwen3_5.py` builds `ArraysCache` for every linear layer. The weights were not downloaded. The serving pin stays Qwen3.

The Gemma 4 revisions downloaded for the previous check were removed from the Hugging Face cache: `gemma-4-e4b-it-4bit` at `475b9088d297` and `deb1db712068`, and `gemma-4-E4B-it-assistant-bf16` at `844e008e06ef`. The cache scan reported 5.371 GB freed.
