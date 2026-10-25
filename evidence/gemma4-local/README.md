# Gemma 4 local pair, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `mlx==0.32.4`, `mlx-lm==0.32.2`.

The Hugging Face cache already had directories for `mlx-community/gemma-4-e4b-it-4bit` and `mlx-community/gemma-4-E4B-it-assistant-bf16`. Each snapshot contained the README only, about 8 KB. The weights were downloaded for this check: the 4-bit target at `475b9088d29754a3379866cf5aeb6b41acd313c2` (5.18 GB) and the assistant at `844e008e06ef5562bdb89428d851d0634d119dcd` (0.19 GB).

The assistant config says `model_type` `gemma4_assistant`. Its model card describes a 4-layer multi-token-prediction drafter for `mlx_vlm`, paired with `mlx-community/gemma-4-E4B-it-bf16`. That bf16 target is 15.91 GB. After the 4-bit download this volume had 7.1 GiB free, so the bf16 target was not downloaded. `mlx_vlm` is not installed. Installing `mlx-vlm==0.7.10` would downgrade `fsspec` and add 42 packages, so it was left out of this environment.

**Measured:** `mlx_lm.load` of the assistant raises `ValueError: Model type gemma4_assistant not supported.` The 4-bit target does load. On the raw prompt `Reply with one word: ready`, 32 tokens, the second pass decoded at 33.93 tok/s. The same call on the pinned Qwen3 1.7B 4-bit model decoded at 94.46 tok/s on its second pass. Neither call applied a chat template. The Gemma text repeated `word: ready`. This is one prompt, not the depth sweep. The serving pin stays Qwen3.
