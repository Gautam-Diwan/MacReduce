# Remote federation setup validation — November 12, 2026

**Measured locally / exact scope:** one Apple M4 MacBook Air (`Mac16,12`), 24 GiB, macOS 26.5. This validates software integration, numerical resume behavior, CPU loopback federation, and local GPU/combined forward-backward preflight. It is not a measurement of three physical Macs, WAN throughput, M1 memory fit, independent ANE clients, full three-epoch convergence, or GPU/ANE simultaneous execution.

## Retained checks

- [Default suite](tests.txt): **106 passed, 8 skipped**. The skips are opt-in hardware tests; unavailable or unexecuted hardware is not counted as a pass.
- [Targeted hardware suite](hardware-tests.txt): **3 passed** with `MACREDUCE_TEST_MLX=1`. A synthetic, unquantized one-block Qwen3 (width 64, intermediate 128, four heads, vocabulary 128, sequence cap 16) tests persistent adapter/AdamW/sampler state across a changed broadcast. GPU covers microbatch/accumulation 1/1 and 2/2; the combined ANE suffix covers 1/1. Dispatch-floor warnings remain in the raw output; these tiny shapes are not throughput evidence.
- [Pinned GPU preflight](gpu-preflight.json), [raw output](gpu-preflight.txt): real forward and backward of cached `mlx-community/Qwen3-0.6B-4bit` at revision `73e3e38d981303bc594367cd910ea6eb48349da8`, rank 8, scale 2, sequence cap 512, first local training-prefix example. Passed without an optimizer update. `HF_HUB_OFFLINE=1` prevented new model downloads during this check.
- [Pinned combined preflight](gpu_ane-preflight.json), [raw output](gpu_ane-preflight.txt): same real pinned model, with the existing one-block ANE suffix, device mask `0x4`, one forward and one backward call recorded. This is the code's program-mask/call gate, not a new independent power/instrumentation attribution experiment. No GPU substitution was used for a failed engine.
- [Ruff](ruff.txt): passed. Immutable Python source snapshots under `raw/sources` are excluded as research data rather than rewritten for style.
- [Dependency lock](lock.txt): passed, including `flwr==1.43.0`.
- [Environment](environment.json): explicit software/OS/CPU/RAM inventory without hostnames, IPs or secrets.

The default suite's Flower integration tests use real loopback gRPC with injected deterministic CPU numerical clients. Three GPU identities, three combined identities and six split identities each complete two concurrent-client rounds. These are transport/cohort tests, not GPU or ANE execution measurements. Tests also cover HMAC tampering, bounded/finite float32 tensors, malicious NPY headers before allocation, disjoint shards, sampler continuation, missing-client rejection, cached replay, changed run/data identity, global-commit failure/restart, failed local-round rollback, and explicit objective weighting.

## Commands

```sh
uv sync --extra cpu --extra mlx --extra serve --extra ane --extra dev --extra federation
python -m pytest -q
ruff check .
uv lock --check
MACREDUCE_TEST_MLX=1 python -m pytest -q tests/test_remote_federation.py::test_resident_mlx_optimizer_and_broadcast_resume
HF_HUB_OFFLINE=1 python -m macreduce federation check --mode gpu --client m4-air-gpu --execute
HF_HUB_OFFLINE=1 python -m macreduce federation check --mode gpu_ane --client m4-air-gpu_ane --execute
```

The checked-in config is a short connectivity pilot, not a calibrated study recipe. Code implementation hashes appear in the preflight plans; they identify that inspected implementation, not a committed Git revision. Later code edits require fresh preflights for the actual run.

## Failures found and recovery

The initial hardware resume test failed with `KeyError: embed_tokens`: flattening optimizer state omitted empty frozen-module containers. The new resident backend now restores saved arrays and calls the public optimizer initialization API to rebuild empty containers while retaining moments; the retained targeted suite passes after that fix.

An initial threaded Flower wrapper test failed to assemble the cohort and hung during shutdown. Stack inspection found unconditional analytics threads invoking `platform.architecture()` subprocess/fork work during active gRPC on macOS. Disabling analytics sending alone did not prevent those threads. The pinned adapter now uses Flower's core resident-client session directly, bypassing the deprecated public wrapper and its telemetry hooks; the retained loopback suite completes after the change. This is a documented version-specific compatibility boundary, not an upstream code patch.

Local Tailscale status was Running/online with no listed peers. No tailnet membership, grants/firewall policies, secrets or user account state were changed. The other two Macs and the actual cross-network training session still need the [runbook](../../wiki/guides/tailscale-federation.md). Standalone ANE training remains unavailable and split execution is refused before a partial cohort starts.
