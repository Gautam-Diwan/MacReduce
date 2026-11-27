---
title: "Run three-Mac federation over Tailscale with Flower"
type: implementation
status: active
updated: 2026-11-25
sources: ["S166", "S175", "S181", "S182", "S183"]
---

# Run three-Mac federation over Tailscale with Flower

The M4 Air 24 GB runs a CPU NumPy coordinator **and a separate local training process**. The M4 Pro 16 GB and M1 Air run one training process each. No AWS instance or fourth machine is used. Tailscale provides the encrypted network; Flower 1.43.0 provides gRPC dispatch/array transport. MacReduce supplies the trainer, signed logical identities, fixed-cohort reduction and persistent recovery state. Config `coordinator: m4-air` records this host; it is a machine label, not a Tailscale hostname.

**Scope:** this is an explicit fixed-round connectivity/trainer pilot, not the finalized proposal experiment. The [hardware batch/accumulation, data allocation, schedule and synchronization calibration is pinned as a TODO](../project/roadmap.md). The example uses three rounds and two local optimizer updates. A [physical GPU pilot on 2026-11-24](../experiments/2026-11-24-three-mac-gpu-federation.md) completed that example on the three Macs. It does not claim three epochs, validation perplexity, or a speedup.

## Backend and cohort support

- `--mode gpu`: three logical clients, one GPU per Mac. Each client must pass a real forward/backward preflight.
- `--mode gpu_ane`: three combined clients using the existing GPU prefix plus ANE decoder suffix. All share the same adapter schema; a suffix compile or execution failure aborts that client, without falling back to GPU. This is mixed placement, not proof of simultaneous execution or a standalone ANE trainer.
- `--mode split`: six distinct GPU/ANE identities, two per Mac. Planning and generic transport tests cover this layout. **Actual training is blocked:** a standalone ANE Qwen3 trainer is not implemented. Both coordinator and worker commands reject the incomplete mode before starting a partial cohort. GPU+ANE does not stand in for standalone ANE.

List identities and settings without loading models:

```sh
uv run macreduce federation plan --mode gpu
uv run macreduce federation plan --mode gpu_ane
uv run macreduce federation plan --mode split
```

Per-machine `microbatch`, `accumulation`, `grad_checkpoint` and objective `weight` are explicit in [configs/federation.yaml](../../configs/federation.yaml). A machine can have nested `gpu`, `gpu_ane`, or `ane` overrides. Do not infer statistical weights from GPU speed. The example's uniform settings are a transport smoke control; calibration is deferred. Each process owns one resident model, local optimizer and deterministic sampler. Split mode will require two independent processes per Mac when its missing backend exists.

## What each teammate must do

1. Install/connect Tailscale on each Mac. Invite all three teammates into one authorized tailnet (each can use their own account), approve devices as required, and keep Tailscale connected. The local Mac was already Running/online when inspected; no peers were listed. No account or tailnet permissions were changed by this work.
2. Check the M4 Air's actual Tailscale IPv4/MagicDNS name. Logical IDs such as `m4-air-gpu` are config labels, not automatically network hostnames. Allow the Pro and the M1 to reach **TCP 8765 on the M4 Air** in your tailnet policy and macOS firewall. No public router forwarding, exit node, cloud relay you manage, Tailscale Funnel, or SSH server is required.
3. Use the same code/config and lockfile on all three machines. If changing config/mode, use a new run directory everywhere. The plan includes a code implementation hash, pinned model/data revisions and complete client settings; mismatches are rejected.
4. Share one random `MACREDUCE_TOKEN` securely between the three machines. Copy [`.env.example`](../../.env.example) to `.env` in the project root and fill in the token. Do not put it in YAML, Git, manifests, command-line arguments or chat. Do not send it to an assistant. `macreduce` loads `.env` automatically; a shell-exported token still wins.
5. Keep each machine plugged in and awake during its run. In split mode, memory budgets for two trainers would share the same physical RAM; no claim of an 8 GB M1 fit is made.

The official CLI documents `tailscale ip`, `status`, `ping` and `netcheck`; capture direct versus relayed paths when making later performance claims. [S166](../sources/S166.md), [S181](../sources/S181.md)

```sh
# On the M4 Air: find its overlay IP. On the Pro and M1: test that IP/name.
tailscale ip -4
tailscale status
tailscale ping AIR_TAILSCALE_IP_OR_NAME
tailscale netcheck
```

## Install and prepare on all three Macs

From the same MacReduce checkout:

```sh
uv sync --extra cpu --extra mlx --extra serve --extra ane --extra dev --extra federation
uv run macreduce federation prepare
```

Preparation uses the existing pinned WikiText preparation code. Each machine caches the same corpus/model locally; the network carries adapter deltas rather than repeatedly shuffling training text or base weights. The actual normalized training-prefix hash must match the coordinator. First use of a backend may download the pinned model and compile kernels; allow for that before starting a timed experiment.

Set the shared token in `.env` on each Mac. For example, generate a token on the M4 Air with Python's `secrets.token_urlsafe(32)`, share it outside Git, and put the same `MACREDUCE_TOKEN=` line in every checkout's `.env`. The commands below load that file; they never print the token.

## Preflight hardware before joining

Run the command matching each machine's logical ID:

```sh
# Pro
uv run macreduce federation check --mode gpu --client m4-pro-gpu --execute
# M4 Air
uv run macreduce federation check --mode gpu --client m4-air-gpu --execute
# M1 Air
uv run macreduce federation check --mode gpu --client m1-air-gpu --execute
```

For combined mode, replace `--mode gpu` with `--mode gpu_ane` and the final `-gpu` client suffix with `-gpu_ane`. A passed combined check requires ANE forward and backward counters to advance under the existing device-mask gate. Run this independently on every actual Mac/OS. A local M4 test does not validate M1 compatibility. This checks finite execution, not a full numerical-quality or power-attribution study.

## Start a GPU connectivity pilot

Start the M4 Air's coordinator first and wait for its **Flower coordinator listening** message. It bootstraps a common adapter on GPU once, then releases the base model; reduction itself is NumPy on CPU. The Air's training client is a separate terminal/process.

```sh
# M4 Air, coordinator terminal (.env contains MACREDUCE_TOKEN)
uv run macreduce federation coordinator --mode gpu --host tailscale \
  --output runs/federation-gpu/coordinator
```

In the following commands replace `AIR_TAILSCALE_IP_OR_NAME` with the M4 Air's real overlay address or MagicDNS name. The same address works for its own local worker.

```sh
# M4 Air, second terminal
uv run macreduce federation worker --mode gpu \
  --server AIR_TAILSCALE_IP_OR_NAME:8765 --client m4-air-gpu \
  --output runs/federation-gpu/m4-air-gpu

# M4 Pro
uv run macreduce federation worker --mode gpu \
  --server AIR_TAILSCALE_IP_OR_NAME:8765 --client m4-pro-gpu \
  --output runs/federation-gpu/m4-pro-gpu

# M1 Air
uv run macreduce federation worker --mode gpu \
  --server AIR_TAILSCALE_IP_OR_NAME:8765 --client m1-air-gpu \
  --output runs/federation-gpu/m1-air-gpu
```

A lightweight network check is also available after the coordinator is listening:

```sh
uv run macreduce federation check --mode gpu --server AIR_TAILSCALE_IP_OR_NAME:8765
```

This check establishes TCP reachability only; signed plan/data/backend admission happens when the worker joins. Clients train concurrently through Flower. A commit requires exactly all expected authenticated logical IDs. Missing, duplicate, stale, malformed or failed results do not produce a smaller-cohort average. The example uses explicit objective weights and does not claim that factor averaging equals dense-update averaging or centralized AdamW.

For a combined pilot, use a **new** root such as `runs/federation-gpu-ane/`, pass `--mode gpu_ane` in every terminal, and use `m4-pro-gpu_ane`, `m4-air-gpu_ane`, `m1-air-gpu_ane`. Prepare/check every combined backend before joining. Leave the old `macreduce federate --execute` command to its existing one-Mac experimental behavior; the new `macreduce federation ...` commands are the remote path.

Optional: an SSH local forward to an M4 Air coordinator bound to `127.0.0.1` can carry the same gRPC port. In that case workers use `127.0.0.1:LOCAL_PORT`. Plain gRPC binds/resolves only to loopback or Tailscale IPv4, not public/wildcard interfaces. Tailnet transport encrypts traffic and application HMAC authenticates envelopes/checksums; this compatibility protocol is not claiming mutual TLS.

## Recovery and outputs

The coordinator writes `state.json` and immutable per-version `adapter-*.npz` files, publishing the state pointer last. Its final adapter is the global model adapter. Each client writes `worker.json` plus an initial checkpoint and round generations containing its adapter, AdamW moments, update counter, sampler cursor and exact prepared outgoing delta. Client files are local training/recovery state, not a substitute for the coordinator's final aggregate.

A lost response or coordinator restart reuses the same cached delta for the same run/version/start hash. A failed local training/preparation attempt restores the previous local trainer state. A client missing optimizer history for an advanced global version is rejected rather than silently reset. This is process-restart recovery, not a claim of power-loss durability or exactly-once physical network delivery.

After a coordinator failure, preserve **every** participant's directories, restart the coordinator with the original config/mode/output plus `--resume`, then restart workers with their original outputs. Do not run two processes against the same worker directory. Worker reconnect attempts are bounded; when exhausted, fix connectivity and rerun the same command. An interrupted attempt may need explicit restarts of all participants; no automatic leader election or quorum reduction is implemented.

```sh
uv run macreduce federation coordinator --mode gpu --host tailscale \
  --output runs/federation-gpu/coordinator --resume
```

Only serialized tensor byte counts are currently recorded in commit receipts. Full gRPC/WireGuard packet overhead, retransmissions, energy and physical-network throughput are unknown, not zero. This pilot does not yet run centralized validation/test or reproduce the full proposal's data-progress schedule. All files under `runs/` are git-ignored.

## Why Flower, and what remains project-owned

The independent library review found official Flower MLX examples and a working resident-client compatibility route. We reuse Flower's transport and parallel RPC scheduling rather than another custom HTTP service. Modern ClientApp lifecycle does not preserve a resident MLX model across messages, and context state is not durable restart storage. The pinned compatibility API preserves the supplied client instance. MacReduce owns authentication, exact cohort policy, array validation and atomic model/optimizer recovery. [S182](../sources/S182.md), [S183](../sources/S183.md)

This route uses two pinned compatibility internals (client session and gRPC lifecycle). Upstream's public wrapper is deprecated and starts telemetry threads even with sending disabled; its macOS platform subprocesses caused an observed threaded-gRPC startup failure during validation. The adapter bypasses that wrapper rather than monkey-patching upstream code. Version checks and integration tests guard the boundary. Migration to a supported resident-worker deployment architecture remains a maintenance TODO; do not upgrade Flower independently on one Mac.

See the [2026-11-24 physical GPU pilot](../../evidence/tailscale-federation-2026-11-24/README.md), the [2026-11-12 one-Mac loopback/preflight record](../../evidence/tailscale-federation-2026-11-12/README.md), [current state](../project/state.md), [transport guide](transport.md) and [deferred study design](../research/three-mac-wireless-federation.md).
