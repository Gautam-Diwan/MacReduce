# Greedy equivalence, 2026-10-25

One M4. `mlx==0.32.4`, `mlx-lm==0.32.2`. Command:

```sh
.venv/bin/python -m macreduce equivalence --max-tokens 16 --output evidence/serve-equivalence/report.json
```

Target `mlx-community/Qwen2.5-1.5B-Instruct-4bit` at `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`. Draft `mlx-community/Qwen2.5-0.5B-Instruct-4bit` at `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`. Those are the Hugging Face `main` revisions queried after the run. The loader did not pass a revision argument; both repos were last modified 2024-09-18, so the download matches those commits if the files were not rewritten without a commit.

**Measured:** greedy speculative decoding at depth 2 matched greedy decoding for 16 tokens on the short, long, and code prompts. All three generations hit the token cap, so this does not exercise an early stop. It is not a timing result, not a depth sweep, and not evidence about CPU or ANE drafters.
