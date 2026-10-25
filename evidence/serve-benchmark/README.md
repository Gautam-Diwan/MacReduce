# Placement study, 2026-10-25

One M4. `mlx==0.32.4`, `mlx-lm==0.32.2`. Target `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. Draft `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Command:

```sh
.venv/bin/python -m macreduce bench --output evidence/serve-benchmark --cycles 3 --max-tokens 32
```

Each trial was a new interpreter. The first replicate of each setting fit the cost model. The later replicates were scored. `report.json` is the machine summary. Per-trial JSON files keep the token ids.

## What was measured

Greedy speculative decoding matched greedy decoding at depths 0, 1, 2, 4, and 8 on the short, long, and code prompts. Three replicates each. All 45 trials finished because they hit 32 tokens, not because the model stopped. An early-stop case was not observed.

Median decode rate and time to first token:

| Segment | Fastest decode | Decode tok/s | No-draft tok/s | Fastest first token |
|---|---|---|---|---|
| short | depth 0 | 97.1 | 97.1 | depth 0, 0.032 s |
| long | depth 4 | 136.9 | 96.1 | depth 0, 1.33 s |
| code | depth 2 | 146.9 | 103.0 | depth 0, 0.115 s |

The cost model selected depth 0, 4, and 4. That matches the decode hindsight on short and long. On code it missed depth 2, and the evaluation passes at depth 4 were about 15 tok/s slower. No segment improved both time to first token and decode rate. The pre-registered joint hypothesis was not met. Where the choice matched the hindsight depth, the comparison is a tie, not a win over that depth.

A CPU-loaded draft on the short prompt matched tokens. Its decode rate was 87.8 tok/s and its time to first token was 0.082 s, both worse than no draft. The draft weights were created on `mx.cpu`. Stock MLX-LM still ran one stream. This is not evidence that the CPU and GPU computed concurrently.

The long prefix was 1,313 tokens. Rebuilding it took 1.32 s. Saving it and loading it back produced the same 4 continuation tokens as recomputing the prefix. The load took 0.90 ms and the file is 44,048,175 bytes. That read happened immediately after the write, so it is a warm page-cache measurement. It is not a cold SSD rate and not a network rate. Applying that one load time to the short and code decisions is what made the prefix rule say "disk" for every segment.

## What was not run

The peer arm needs a second Mac. The ANE gate needs `coremltools` and a conversion of the draft model. Neither was available. No ANE power counter was read. Per-step changes of the draft depth, inside one generation, were not implemented. The reported adaptive policy picks one depth per segment before the evaluation passes.
