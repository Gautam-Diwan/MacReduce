# MacReduce

MacReduce is a research prototype for federated training across heterogeneous Macs. Each worker holds a full model, trains locally, and sends LoRA adapter updates to a coordinator. The [three-Mac setup below](#train-across-different-networks) uses Flower over Tailscale, with the M4 Air 24 GB hosting both the coordinator and its own worker.

The course proposal compares GPU clients, combined GPU+Neural Engine clients, and separate GPU and Neural Engine clients. Remote GPU and combined training are implemented. Separate ANE training remains blocked by a missing backend. Hardware batch sizes and synchronization settings still need calibration. The earlier 29,888-parameter synthetic trainer remains available as a separate demonstration.

## Research wiki

Start with the [knowledge index](wiki/index.md), [implementation roadmap](wiki/project/roadmap.md), and [research contribution candidates](wiki/research/novelty.md). The wiki includes source snapshots, notes with reading scope, implementation gates, experiment plans, and maintenance instructions. The active [proposal](docs/proposal.md) is retained alongside historical artifacts under `docs/archive/`.

```sh
python tools/wiki.py search 'ANE optimizer'
python tools/wiki.py lint
```

See [how to maintain the wiki](wiki/how-to-use.md) for source intake and agent prompts. The project now lives in the course folder under `Project/MacReduce`; historical evidence retains the original project name.

## What has been verified

Two independent CPU worker processes completed eight rounds over real loopback HTTP with batch sizes 2 and 4. Fixed synthetic validation loss decreased from **6.0587 to 1.7681**. Tests verify the pooled-SGD numerical identity at one local step, causal masking, weighted reduction, checkpoint validation, recovery, authentication, and idempotent retries after a lost response.

On 2026-10-24 the same M4 passed the Metal smoke check, optional MLX parity tests, and a three-round GPU loopback demo. Validation loss fell from 6.058713912963867 to 3.713829755783081. That demo used one machine. `xctrace` was unavailable, so no Metal capture exists for it. The CPU operator trace and Apple Accelerate matrix probe ran. See [evidence/mlx-validation](evidence/mlx-validation/README.md).

On 2026-11-12 the default test suite passed 106 tests with eight optional skips. Three targeted hardware recovery tests passed on the local M4 Air 24 GB. The pinned Qwen3 model passed GPU and combined GPU+ANE forward/backward preflights there. Loopback transport tests covered three and six simulated clients. A physical three-Mac run and execution on the M1 remain untested. See the [federation validation record](evidence/tailscale-federation-2026-11-12/README.md).

See [evidence](evidence/README.md) for measured results and [the profiling guide](docs/profiling.md) for measurement methods and limitations.

## Install on each Mac

Use Python 3.12 or newer on Apple Silicon for MLX and Neural Engine work. [uv](https://docs.astral.sh/uv/) is the supported installer; it creates `.venv` and installs from the pinned `uv.lock`.

Install uv if needed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From this folder:

```sh
uv sync --extra cpu --extra mlx --extra serve --extra ane --extra dev --extra federation
uv run macreduce doctor --smoke --output runs/hardware.json
uv run pytest -q
```

`cpu` is the PyTorch CPU reference backend. `mlx` pins MLX on arm64 macOS. `serve` pins `mlx-lm` and `coremltools`. `ane` pins `aneforge` for combined GPU+ANE training. `dev` adds pytest and ruff. `federation` pins Flower 1.43.0 for remote workers. Keep the lockfile and dependencies identical across all three Macs.

Use `uv run` before commands (`uv run python main.py ...`, `uv run macreduce demo ...`) or activate the environment:

```sh
source .venv/bin/activate
macreduce doctor --smoke
```

Without uv, use a normal venv and pip (no lockfile):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu,mlx,serve,ane,dev,federation]'
```

`serve` pins `mlx-lm==0.32.2` and `coremltools==9.0`. Placement math does not need either. `macreduce equivalence` downloads the pinned Qwen3 1.7B and 0.6B 4-bit MLX models on first use. `macreduce ane` converts a small convolution and reads its compute plan. `macreduce ane-workload` times one decoder block on CPU, GPU, and the Neural Engine and ranks those measured times. `macreduce infer` times a 28-layer forward at the Qwen3-0.6B widths on those same devices, then generates with the pinned Qwen3 pair. Train from the repo root with a YAML config:

```sh
python main.py --config configs/qlora_m4.yaml
```

`training.batch_size: auto` measures steady tokens per second and keeps the fastest microbatch that fits in the GPU's recommended working set (`infrastructure.memory_limit_bytes: auto`). `grad_checkpoint: auto` recomputes activations only when the plain step does not fit. `placement.mode: gpu_ane` keeps the early 4-bit Qwen blocks on the GPU and runs a suffix of full decoder blocks, frozen base plus LoRA, as one Neural Engine program. The run is three epochs on the first 4,096 WikiText-2 lines, with a tqdm step bar. Each launch creates a timestamped directory under `logging.dir` and writes `run.log` and `figures/` only there. Checkpoints under that directory's `checkpoints/` keep the adapter, AdamW moments, and the step every epoch. Resume with `--resume latest`. The run writes mactop power and memory, plus matplotlib figures for loss, perplexity, instantaneous GPU and Neural Engine power, memory, and latency. `macreduce figures <run-directory>` redraws those files. `macreduce federate` lists the federated cohorts and checks without training them. `macreduce train-plan` keeps the earlier all-reduce list. The Qwen2.5 study remains in `evidence/serve-benchmark`.

```sh
macreduce place --acceptance 0.9 --prefix-tokens 4096 --prefill-tokens-per-second 2000 --kv-bytes 50000000 --disk-bytes-per-second 200000000
macreduce trace
macreduce equivalence --max-tokens 32 --output runs/equivalence.json
macreduce model-check --output runs/model-check.json
```

The `cpu` extra installs the explicit PyTorch CPU reference backend. The `mlx` extra installs MLX on Apple silicon. GPU access must be available to the terminal process. There is no silent backend fallback. `requirements-tested.txt` records the exact versions used for the delivered CPU tests; the project extras allow a broader development range.

Before mixing MLX and CPU workers, run the numerical parity check from a normal Mac terminal:

```sh
MACREDUCE_TEST_MLX=1 pytest -q tests/test_mlx_optional.py
```

This check is skipped by default. When explicitly enabled, an unavailable GPU causes a failure rather than a false pass.

## Train across different networks

This setup uses only your three Macs. They can share a network or connect from separate networks. The M4 Air 24 GB needs two terminals. The Pro and the M1 each need one.

| Machine | RAM | Processes | GPU client ID |
|---|---|---|---|
| M4 MacBook Air | 24 GB | Coordinator and local worker | `m4-air-gpu` |
| M4 MacBook Pro | 16 GB | Worker | `m4-pro-gpu` |
| M1 MacBook Air | 8 GB | Worker | `m1-air-gpu` |

These client IDs are config labels. They do not create Tailscale hostnames.

### 1. Connect the Macs with Tailscale

Install Tailscale on every Mac and join one authorized tailnet. Invite your teammates with their own accounts and approve their devices if your tailnet requires it. Keep the machines plugged in and awake.

On the M4 Air 24 GB, get its Tailscale IPv4 address:

```sh
tailscale ip -4
tailscale status
```

On the Pro and the M1, replace `AIR_TAILSCALE_IP_OR_NAME` with that address or the Air's MagicDNS name:

```sh
tailscale ping AIR_TAILSCALE_IP_OR_NAME
```

Allow the Pro and the M1 to reach TCP port 8765 on the M4 Air in your tailnet policy and macOS firewall. The coordinator uses Tailscale's encrypted connection. You do not need router port forwarding, an SSH server, or an AWS coordinator.

### 2. Install and prepare on every Mac

Use the same MacReduce source, `configs/federation.yaml`, and `uv.lock` on all three machines. Run these commands from the project root after installing uv as described above:

```sh
uv sync --extra cpu --extra mlx --extra serve --extra ane --extra dev --extra federation
uv run macreduce federation plan --mode gpu
uv run macreduce federation prepare
```

`prepare` downloads the pinned WikiText-2 data. Each Mac downloads and caches the pinned Qwen3-0.6B 4-bit base model on its first hardware check. Training exchanges adapter updates. It does not move the corpus or base model between Macs each round.

The [default config](configs/federation.yaml) runs three rounds with two local optimizer updates per round. It uses microbatch 1 and accumulation 1 on every machine, with gradient checkpointing on the M1. These are connectivity test settings. Per-machine batches, data allocation, objective weights, and sync frequency remain [pinned TODOs](wiki/project/roadmap.md) for the study.

### 3. Put the shared token in `.env` on every Mac

Copy the example file at the project root, then put the same token on all three machines:

```sh
cp .env.example .env
uv run python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Paste the generated value into `MACREDUCE_TOKEN=` in `.env`. Share that file contents with your teammates through a private channel, not Git or chat with an assistant. `.env` is gitignored. `macreduce` loads it from the project root automatically. An already-exported `MACREDUCE_TOKEN` in the shell still wins.

### 4. Check each machine's GPU

Run only the command for the Mac you are using:

```sh
# M4 Pro
uv run macreduce federation check --mode gpu --client m4-pro-gpu --execute

# M4 Air
uv run macreduce federation check --mode gpu --client m4-air-gpu --execute

# M1 Air
uv run macreduce federation check --mode gpu --client m1-air-gpu --execute
```

Each check must report `forward_backward_preflight` as `passed`. Resolve failures before starting the cohort. The first check may take longer while the model downloads. A check on one Mac does not establish compatibility or memory fit on the others.

### 5. Start the M4 Air's coordinator

In the M4 Air's first terminal, with the shared token exported:

```sh
uv run macreduce federation coordinator --mode gpu --host tailscale \
  --output runs/federation-gpu/coordinator
```

Wait for `Flower coordinator listening`. Startup initializes a common adapter on the Air's GPU, then releases the base model. The coordinator averages updates on CPU. It waits for all three workers before training.

### 6. Start one worker on each Mac

Replace `AIR_TAILSCALE_IP_OR_NAME` in all commands with the M4 Air's real Tailscale address or MagicDNS name. Use `HOST:8765` without an `http://` or `https://` prefix. The Air's worker uses the same address as the Pro and the M1.

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

All three workers must join with the same mode, plan, data hash, and token. A round commits only after all expected clients succeed. Missing or failed clients stop the run rather than produce a partial average.

If a worker cannot connect, run this after the coordinator starts:

```sh
uv run macreduce federation check --mode gpu \
  --server AIR_TAILSCALE_IP_OR_NAME:8765
```

This checks TCP reachability. Worker admission checks the token, plan, and data separately. For a failed connection, check `tailscale status`, `tailscale ping`, the Air's address, and TCP 8765 permissions. `tailscale netcheck` can help diagnose a relayed connection.

### Use GPU+ANE or inspect split mode

For combined GPU+ANE training, change every command's mode to `--mode gpu_ane`. Use client IDs `m4-pro-gpu_ane`, `m4-air-gpu_ane`, and `m1-air-gpu_ane`. Run the hardware checks again on all three Macs. Use a new output root, such as `runs/federation-gpu-ane`, for both coordinator and workers.

Combined mode places an early model prefix on GPU and a decoder suffix on ANE. ANE forward and backward checks must pass. A failure stops that client without a GPU fallback. This mode does not establish simultaneous GPU and ANE execution.

```sh
# Example combined preflight on the M1
uv run macreduce federation check --mode gpu_ane --client m1-air-gpu_ane --execute

# Inspect the proposed six-client layout without loading a model
uv run macreduce federation plan --mode split
```

Split mode assigns separate GPU and ANE clients to each Mac. Its standalone ANE Qwen3 trainer is missing. Coordinator and worker startup reject this mode. Six-client planning and transport tests do not establish six-device training support.

### Resume an interrupted run

Keep the coordinator directory on the M4 Air and each worker's directory on its own Mac. After fixing the failure, restart the coordinator with the original settings and `--resume`:

```sh
uv run macreduce federation coordinator --mode gpu --host tailscale \
  --output runs/federation-gpu/coordinator --resume
```

Restart all workers with their original commands and output directories. Export the shared token again in any new terminal. Workers retain their adapter, optimizer moments, sampler position, and pending update. Do not delete a worker's state or run two processes against its directory. Reconnect attempts are bounded, so a long outage can require these manual restarts.

For a fresh experiment or a config change, choose new output directories for every participant. The coordinator's `state.json` identifies the committed global `adapter-*.npz`. Worker checkpoints contain local recovery state. All `runs/` outputs are git-ignored.

This pilot does not yet reproduce the proposal's three-epoch curve or run centralized validation and test evaluation. See the [full runbook](wiki/guides/tailscale-federation.md) for recovery semantics, transport limits, and the Flower dependency boundary. `macreduce federation ...` is the remote QLoRA command family. `macreduce federate --execute` is an earlier one-Mac experiment.

## Run the local demonstration

```sh
macreduce demo --backend torch-cpu --output runs/cpu-demo
macreduce report runs/cpu-demo
```

This starts a coordinator and two worker processes. The workers use different batch sizes, a shared initial checkpoint, SGD without momentum, ten local steps per round, and eight averaging rounds. Output includes `result.json`, `state.json`, checkpoints, `events/*.jsonl`, `report.md`, `summary.json`, and per-node trace files. Use a new output directory for each demo.

To try MLX after the parity check passes:

```sh
macreduce demo --backend mlx-gpu --output runs/mlx-demo
```

## Run the synthetic demo on two Macs over Wi-Fi

Install the same source version and dependencies on both machines. The coordinator can also run a worker. These commands run the synthetic demonstration workload; both workers must connect before training begins.

1. Generate a secret on the coordinator and set the same `MACREDUCE_TOKEN` environment variable in every participating terminal. Keep it out of logs and source control.

```sh
export MACREDUCE_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

2. Start the coordinator, using explicit worker names:

```sh
macreduce coordinator --host 0.0.0.0 --port 8765 \
  --nodes m4,second --steps 10 --rounds 8 --round-timeout 180 \
  --output runs/wifi
```

3. In another terminal on the coordinator Mac, set the same token and run:

```sh
macreduce worker --url http://127.0.0.1:8765 --node m4 \
  --backend mlx-gpu --batch-size 4 --output runs/wifi/events
```

4. On the second Mac, set the same token. Replace `COORDINATOR_IP` with the coordinator's Wi-Fi address shown in macOS Network settings:

```sh
macreduce worker --url http://COORDINATOR_IP:8765 --node second \
  --backend mlx-gpu --batch-size 2 --output runs/wifi-worker/events
```

Choose `torch-cpu` explicitly for a CPU worker. Start with a small batch on a memory-constrained Mac. Different M-series generations and memory sizes can participate; physical heterogeneity still needs experimental validation. An unavailable selected backend stops the worker. A firewall or access point with client isolation can prevent connections even when both Macs use the same Wi-Fi name.

The example HTTP transport is for a trusted private lab network and synthetic data. It authenticates but does not encrypt requests. For encrypted transport, configure the coordinator with `--tls-cert` and `--tls-key`, use an `https://` URL, and give workers `--ca-file` for your certificate authority. Alternatively, leave the coordinator bound to localhost and use SSH port forwarding over Wi-Fi. Do not publish the prototype's development HTTP server to the internet. TLS configuration is implemented but not part of the delivered test evidence.

After the run, copy the second worker's event file into `runs/wifi/events/second.jsonl` on the coordinator, then run:

```sh
macreduce report runs/wifi
```

Use unique node names and one run per report directory. Node clocks are not assumed synchronized.

## Observe a synthetic demo run

This section describes the synthetic demo's HTTP coordinator. The Flower federation coordinator uses the commands and output files described above.

`GET /status` returns current version, worker membership, pending submissions, validation loss, and any failure. `GET /metrics` exposes Prometheus gauges for version, pending workers, validation loss, and failure. Both require the same bearer token:

```sh
curl -H "Authorization: Bearer $MACREDUCE_TOKEN" http://127.0.0.1:8765/status
```

The coordinator stays up after training finishes so these endpoints remain available. Stop it with Control-C. Detailed timings and step losses are in JSONL files; the initial Prometheus endpoint is a status view, not the complete profiling store.

```sh
# Capture system telemetry in a second terminal while training runs.
macreduce telemetry --seconds 30 --output runs/wifi/telemetry

# Profile one CPU step after warmup in a separate, instrumented run.
macreduce demo --backend torch-cpu --torch-trace --output runs/cpu-profile

# Public CPU matrix multiplication, independent of the trainer.
macreduce matmul --size 512 --repeats 10 --output runs/accelerate.json
```

The [profiling guide](docs/profiling.md) covers Instruments, Metal captures, powermetrics, interpretation of ANE activity, and CPU matrix-unit uncertainty.

## Train the synthetic model on local text

Workers accept `--train-text /path/to/train-shard.txt`; the coordinator accepts `--validation-text /path/to/validation.txt`. Each must contain at least 1,024 bytes. The tokenizer is fixed UTF-8 bytes, the sequence length is 32, and the model remains the small demonstration architecture. Prepare a disjoint validation split yourself. Input text stays on each worker; model updates still travel over the network.

Each worker samples random windows from its local corpus. The synthetic default uses separate generator seeds for training and validation, but combinations can overlap. It is a learning smoke test, not a benchmark of generalization. With different corpus sizes or domains, token weighting gives more influence to workers that process more tokens; it is not automatically equivalent to sampling uniformly from the concatenated corpus.

## Synthetic trainer recovery and algorithm semantics

- Membership, batch sizes, backend choices, data fingerprints, learning rate, and local steps are fixed during a run. Faster workers may have larger configured batches, but automatic calibration is future work.
- Each accepted update is identified by run ID, model version, node, token count, and content checksum. A lost-response retry reuses the exact update bytes. A different update for the same identity is rejected.
- The coordinator stores float32 checkpoints without pickle and validates the embedded model config, parameter names, shapes, dtypes, finiteness, and archive size. Updates are averaged in float64 and returned to float32.
- Workers move checkpoints in chunks. Each artifact has a manifest, a 64 MiB size cap, and per-chunk hashes. A partial or corrupted upload is not averaged. Retrying the same update does not apply it twice.
- Committed checkpoints and receipts survive restart. Pending updates and in-flight chunk uploads are not durable: an interrupted round restarts from its previous checkpoint. Restart the coordinator with the same settings plus `--resume`; reconnect or restart failed workers with the same IDs and settings. An orphan checkpoint written before a failed manifest commit is ignored.
- A round timeout fails the fixed cohort; it does not switch to partial averaging. A worker's connection retries are bounded to approximately 60 seconds, so longer outages may require restarting that worker. A running coordinator serializes update validation and reduction; it is not a high-availability service.
- For K=1, weighted local SGD matches the pooled-gradient update. For K>1 it is local SGD with averaging, not exact synchronous training and not DiLoCo. There is no momentum or optimizer-state merge in v0.1.

## Development layout and next milestones

Install the Git pre-commit hook so a commit is refused unless `ruff check .` and `python -m pytest -q` both pass:

```sh
sh tools/install-git-hooks.sh
```

The hook lives in `.githooks/pre-commit` and uses the project Ruff config and the default pytest suite (MLX tests stay opt-in). `cli.py` is the composition root. Packages are `cluster` (local-SGD prototype), `train` (QLoRA curve), `federation` (Flower/Tailscale), `ane`, `serve`, and `observe`.

The synthetic demo uses token-weighted local SGD on the 29,888-parameter model. The remote QLoRA path is `macreduce.federation`. It keeps each worker's AdamW state across broadcasts and reduces adapter updates with explicit objective weights. Its next milestones are physical three-Mac validation, hardware calibration, the standalone ANE trainer, and the proposal's matched evaluation. See the [roadmap](wiki/project/roadmap.md). Both transports reject artifacts larger than 64 MiB.

The hardware inventory distinguishes M5 GPU neural accelerators from ANE and detects CPU SME/SME2 features where permitted. It also records media engines as a future preprocessing opportunity. Inferred presence never counts as verified execution, and unknown future chip generations are not automatically declared supported.
