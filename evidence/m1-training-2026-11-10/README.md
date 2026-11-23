# M1 training pilots and failed real-model ANE gate

**Result:** GPU QLoRA completed a bounded pilot. The combined GPU+ANE program
also completed, but failed the real-Qwen3 correctness gate and is **not an
accepted training benchmark**. Standalone ANE linear training passed; a
standalone Qwen3 ANE client is still unimplemented. No three-epoch study ran.

Machine: M1 MacBook Air, 8 GiB RAM, macOS 27.2.1, native Python 3.12.0.
Exact package versions and results are in [summary.json](summary.json).
These runs use a project venv with inherited system-site packages, not a full
`uv sync` environment. No system memory/security/fan settings changed.

| Matched pilot | GPU | GPU+ANE |
| --- | ---: | ---: |
| Updates / scored training tokens | 10 / 1,086 | 10 / 1,086 |
| Initial sampled test perplexity | 97.5221 | 97.5221 |
| Final sampled test perplexity | 73.5956 | 112.5949 |
| Peak MLX allocations | 687,620,656 bytes | 1,362,022,595 bytes |
| Median reported step time | 497.46 ms | 627.30 ms |
| Acceptance | Pilot only | Failed correctness |

Both used pinned Qwen3-0.6B 4-bit, rank-8 LoRA/scale 2, sequence 128, batch 1,
checkpointing, seed 0, the first 32 nonempty WikiText training records, and two
fixed evaluation batches. This is not full-split quality or a speedup study.
Steps contain unequal token counts; there is one measured run per arm. Initial
compilation, cache state, and setup affect timings. The hybrid's invalid math
also prevents interpreting its timing as a useful-training comparison.

The new sampler isolates training RNG from backend setup and evaluation. The
training token-count sequences match exactly. Both baseline evaluations use
the same fixed seed and have exactly matching loss. Earlier pilots had an
uncontrolled evaluation/setup RNG and are retained under ignored `runs/`;
their quality numbers are superseded by this matched pair.

The dataset comes directly from commit
`b08601e04326c79dfdd32d625aee71d232d685c3`, with split SHA-256 values checked
against the [pinned file listing](dataset-tree.json). The previous dataset
server helper recorded that revision without requesting it. The code now
verifies the immutable parquet files; PyArrow is included in the serving extra
and lockfile. Old caches lacking verified source hashes are rebuilt.

**Why the ANE result is rejected:** [real-block measurements](qwen-norm-parity.json)
at GPU adapter steps 0, 5, and 10 show forward cosine 0.914–0.920 and all ANE
LoRA gradients zero where the GPU gradients are nonzero. The ANE suffix output
equals its input on these batches. An isolated input RMSNorm program also
returns all zeros, whereas the reference norm is nonzero. Input peaks are
5,760–6,624. This localizes a failing operation on this workload; an overflow
mechanism is a hypothesis, not established by these measurements. The
reference is the GPU quantized block, not a full float64 decoder oracle.

[ANE linear checks](ane-correctness.json) passed loss/gradient/SGD comparisons
with NumPy at scales 0, 0.01, 1, and 4, plus 100 finite fixed-batch updates.
ANE arithmetic uses fp16; the reference uses float64. Tolerances are rtol 0.02,
atol 0.001; raw errors are retained. [The existing tiny decoder test](ane-decoder-tests.txt)
also passed its forward and finite-difference gate. Neither tiny test predicts
the real-model failure. [Adapter checks](adapter-check.json) show finite saved
values; finiteness alone does not establish correctness.

Both successful pilots saved checkpoints at steps 5 and 10. Their state and
weight/optimizer hashes are archived under [gpu](gpu/checkpoint-10.json) and
[gpu_ane](gpu_ane/checkpoint-10.json). Actual adapters remain under ignored
`runs/`. Raw result `status: ok` means execution completed; it does not override
the failed correctness classification here.

System mactop sampling worked, but ANE/CPU/DRAM channels reported zero even
while the explicit ANE programs ran. The summary leaves accepted energy
unavailable (`null`). No energy ranking or proof from power is claimed.
Archived telemetry strips process/application details. Program masks were 0x4.

**Stops and recovery:** the [first GPU attempt](attempts/gpu-attempt-1/exit.json)
stopped for swap growth above 512 MiB before its first update. After the user
saved work and closed heavy apps, training ran under normal memory pressure.
The [first hybrid attempt](attempts/gpu-ane-attempt-1/exit.json) stopped below
the original 15 GiB disk reserve. The user explicitly chose 10 GiB for subsequent
smoke runs. Memory/swap/time limits were unchanged. Both stopped runs were
terminated and reaped by the supervisor; none were silently resumed.

For reproducibility, add the local telemetry binary to PATH and keep caches
under the ignored project run directory:

```sh
export PATH="$PWD/runs/M1_8GB/setup/bin:$PATH"
export PYTHONPATH=src
export HF_HOME="$PWD/runs/M1_8GB/huggingface"
export ANEFORGE_CACHE_DIR="$PWD/runs/M1_8GB/ane-cache"
export MPLCONFIGDIR="$PWD/runs/M1_8GB/matplotlib"
.venv/bin/python tools/bounded_smoke.py --config configs/M1_8GB/gpu_smoke.yaml --output runs/M1_8GB/gpu-next
.venv/bin/python tools/ane_correctness.py --output runs/M1_8GB/ane-correctness-next.json
```

The supervisor stops for critical pressure, <256 MiB available memory, >512 MiB
swap growth, disk below the configured reserve, or 900 seconds. These are
best-effort checks, not OS isolation or a no-crash guarantee. Before cache
redirection, ANEForge created about 139 MiB in its default
`~/Models/.aneforge-cache`; that cache was preserved. Shared-volume free-space
changes cannot be fully attributed to this experiment.

The hybrid smoke config now requires a passing real-block gradient report;
the current failed report blocks ordinary launches. The independent
[split-client gate](separate-client-gate.json) also explicitly rejects Qwen3
execution. A tiny ANE linear result is not substituted for that missing arm.
See [the maintained experiment report](../../wiki/experiments/2026-11-10-m1-training.md)
for the remaining work.

Validation: [114 passed, 5 skipped](tests.txt) in the final default suite;
hardware skips are not passes. The separately enabled tiny ANE decoder suite
passed four tests, including its hardware check. New project code passes Ruff,
wiki lint passes (164 pages, 129 sources), and the updated lockfile passes
`uv lock --check --offline`. Whole-tree Ruff still reports 13 pre-existing
violations in immutable source snapshots. The inherited SciPy/NumPy pair emits
one warning. [The final hybrid preflight](blocked-hybrid-preflight.json) confirms
that the correctness failure blocks training before output allocation.
