# Physical three-Mac GPU federation — 24 November 2026

**Measured / exact scope:** one Flower/Tailscale GPU connectivity pilot on three physical Macs. Coordinator plus `m4-air-gpu` on an M4 MacBook Air 24 GB; `m4-pro-gpu` on an M4 Pro 16 GB; `m1-air-gpu` on an M1 Air 8 GB. Mode `gpu`, pinned `mlx-community/Qwen3-0.6B-4bit` revision `73e3e38d981303bc594367cd910ea6eb48349da8`, rank-8 LoRA, three rounds, two local AdamW steps, sequence 512, first 4,096 WikiText-2 lines, disjoint IID shards. Git tree at analysis: `92b3cff`.

This is **not** a three-epoch curve, validation/test perplexity, speedup, energy, or calibrated heterogeneous study. Combined GPU+ANE and split modes were not run. Adapter `.npz` / `.safetensors` stay in gitignored `runs/`; this folder keeps hashes, receipts, and compact worker metadata.

## Commands

```sh
uv run macreduce federation coordinator --mode gpu --host tailscale \
  --output runs/federation-gpu/coordinator
uv run macreduce federation worker --mode gpu \
  --server TAILSCALE_IPV4:8765 --client m4-air-gpu \
  --output runs/federation-gpu/m4-air-gpu
# Pro: --client m4-pro-gpu ; M1: --client m1-air-gpu
```

The coordinator bound to a Tailscale CGNAT IPv4 (`--host tailscale`). Concrete overlay addresses are omitted here.

## What committed

- `run_id` `ce017ce1-1a68-489a-bc09-a8bcdd299cc2` on coordinator and all three `worker.json` files.
- `committed global version 1/3`, `2/3`, `3/3`; Flower `received 3 results and 0 failures` each round.
- 392 LoRA tensors, 5,046,272 fp32 parameters, ~20.3 MiB per adapter; all coordinator adapters finite.
- Placement in every receipt: `{device: gpu}`. M1 used `grad_checkpoint: true` and still finished these six steps.
- Air worker `prepared_version: 2` with coordinator `version: 3` matches the start-version numbering.
- Flower skipped evaluation. Local train losses are two WikiText lines per client per round, not held-out perplexity.

## Wall clock (Air and coordinator mtimes only)

| Event | Local time 2026-11-24 |
|---|---|
| Bootstrap `adapter-000000.npz` | 22:10:49 |
| Air worker `initial/` | 22:30:22 |
| Commit v1 / v2 / v3 | 22:31:23 / 22:32:18 / 22:33:45 |
| Round barriers | 61 s, 55 s, 87 s |

About twenty minutes elapsed between bootstrap and the first commit while the Pro was offline. Air local steps were 1–4 s; the rest of each round is straggler compute plus transport. M1 and Pro file mtimes on this checkout are copy time (23:03), not training time.

## Tokens and local train loss (not a quality metric)

| Round | M1 tokens / loss | Air tokens / loss | Pro tokens / loss |
|---|---|---|---|
| 1 | 599 / 3.72 | 27 / 7.28 | 148 / 4.07 |
| 2 | 108 / 3.33 | 104 / 4.47 | 248 / 4.05 |
| 3 | 372 / 3.05 | 242 / 3.71 | 83 / 4.50 |

Aggregation weights were 1/3 each after normalize. Equal example counts are not equal token mass.

## Unknown (not zero)

Direct vs relayed Tailscale path; gRPC/WireGuard overhead beyond `serialized_tensor_bytes = 20235264`; M1/Pro GPU step times; energy; memory; M1 fit for 4,096 local steps; `gpu_ane` / `split`.

## Files

- [summary.json](summary.json): plan identity, adapter hashes, Air delta L2, unknown list.
- [receipts.json](receipts.json): per-round per-client tokens, loss, updates, cursor, payload bytes.
- [workers.json](workers.json): compact `worker.json` plus cursor/updates; no optimizer dtypes.
- [coordinator-terminal.txt](coordinator-terminal.txt), [worker-m4-air-terminal.txt](worker-m4-air-terminal.txt): Air terminals with overlay IPs redacted.

Full tensors remain under `runs/federation-gpu/` on the Air. Do not commit them.

The November 12 record is one-Mac loopback and preflight only: [tailscale-federation-2026-11-12](../tailscale-federation-2026-11-12/README.md). Wiki report: [2026-11-24 three-Mac GPU federation](../../wiki/experiments/2026-11-24-three-mac-gpu-federation.md).
