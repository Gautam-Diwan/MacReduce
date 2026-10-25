# MacReduce

A working research prototype for observable local-SGD training across heterogeneous Macs. Workers train locally for several steps, then exchange models through a coordinator. The transport works over ordinary TCP/IP, including Wi-Fi.

**Version 0.1 is a small-model trainer.** It currently trains a 29,888-parameter causal byte Transformer. Every worker holds the entire model. It does not yet train on the ANE, shard large models for inference, or automatically assign work based on hardware. No pretrained weights or external dataset downloads are needed for the demo.

## Research wiki

Start with the [knowledge index](wiki/index.md), [implementation roadmap](wiki/project/roadmap.md), and [research contribution candidates](wiki/research/novelty.md). The wiki includes 22 source snapshots, source notes with reading scope, implementation gates, experiment plans, and maintenance instructions. The active [proposal](docs/proposal.md) is retained alongside historical artifacts under `docs/archive/`.

```sh
python tools/wiki.py search 'ANE optimizer'
python tools/wiki.py lint
```

See [how to maintain the wiki](wiki/how-to-use.md) for source intake and agent prompts. The project now lives in the course folder under `Project/MacReduce`; historical evidence retains the original project name.

## What has been verified

Two independent CPU worker processes completed eight rounds over real loopback HTTP with batch sizes 2 and 4. Fixed synthetic validation loss decreased from **6.0587 to 1.7681**. Tests verify the pooled-SGD numerical identity at one local step, causal masking, weighted reduction, checkpoint validation, recovery, authentication, and idempotent retries after a lost response.

On 2026-10-24 the same M4 ran `macreduce doctor --smoke` (Metal matmul result 4096.0), the opt-in MLX parity test for the default model and a 2-layer config, and a loopback `mlx-gpu` demo of 3 rounds and 4 steps (validation loss 6.058713912963867 to 3.713829755783081). That demo is one machine. There is no Metal capture: `xctrace` was not installed. Wi-Fi, multiple physical Macs, ANE execution, and M5 GPU neural accelerators have not been tested. The CPU operator trace and Apple Accelerate matrix probe did run. See [evidence/mlx-validation](evidence/mlx-validation/README.md).

See [evidence](evidence/README.md) for measured results and [the profiling guide](docs/profiling.md) for measurement methods and limitations.

## Install on each Mac

Use Python 3.12 or newer. From this folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu,mlx,dev]'
macreduce doctor --smoke --output runs/hardware.json
pytest -q
```

If `python3.12 -m venv` fails because a uv-managed interpreter looks for its standard library under `/install`, use `uv venv .venv --python 3.12` and `uv pip install --python .venv/bin/python -e '.[cpu,mlx,dev]'` instead.

The `cpu` extra installs the explicit PyTorch CPU reference backend. The `mlx` extra installs MLX on Apple silicon. GPU access must be available to the terminal process. There is no silent backend fallback. `requirements-tested.txt` records the exact versions used for the delivered CPU tests; the project extras allow a broader development range.

Before mixing MLX and CPU workers, run the numerical parity check from a normal Mac terminal:

```sh
MACREDUCE_TEST_MLX=1 pytest -q tests/test_mlx_optional.py
```

This check is skipped by default. When explicitly enabled, an unavailable GPU causes a failure rather than a false pass.

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

## Connect two Macs over Wi-Fi

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

## Observe a live run

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

## Train on local text

Workers accept `--train-text /path/to/train-shard.txt`; the coordinator accepts `--validation-text /path/to/validation.txt`. Each must contain at least 1,024 bytes. The tokenizer is fixed UTF-8 bytes, the sequence length is 32, and the model remains the small demonstration architecture. Prepare a disjoint validation split yourself. Input text stays on each worker; model updates still travel over the network.

Each worker samples random windows from its local corpus. The synthetic default uses separate generator seeds for training and validation, but combinations can overlap. It is a learning smoke test, not a benchmark of generalization. With different corpus sizes or domains, token weighting gives more influence to workers that process more tokens; it is not automatically equivalent to sampling uniformly from the concatenated corpus.

## Recovery and algorithm semantics

- Membership, batch sizes, backend choices, data fingerprints, learning rate, and local steps are fixed during a run. Faster workers may have larger configured batches, but automatic calibration is future work.
- Each accepted update is identified by run ID, model version, node, token count, and content checksum. A lost-response retry reuses the exact update bytes. A different update for the same identity is rejected.
- The coordinator stores float32 checkpoints without pickle and validates the embedded model config, parameter names, shapes, dtypes, finiteness, and archive size. Updates are averaged in float64 and returned to float32.
- Workers move checkpoints in chunks. Each artifact has a manifest, a 64 MiB size cap, and per-chunk hashes. A partial or corrupted upload is not averaged. Retrying the same update does not apply it twice.
- Committed checkpoints and receipts survive restart. Pending updates and in-flight chunk uploads are not durable: an interrupted round restarts from its previous checkpoint. Restart the coordinator with the same settings plus `--resume`; reconnect or restart failed workers with the same IDs and settings. An orphan checkpoint written before a failed manifest commit is ignored.
- A round timeout fails the fixed cohort; it does not switch to partial averaging. A worker's connection retries are bounded to approximately 60 seconds, so longer outages may require restarting that worker. A running coordinator serializes update validation and reduction; it is not a high-availability service.
- For K=1, weighted local SGD matches the pooled-gradient update. For K>1 it is local SGD with averaging, not exact synchronous training and not DiLoCo. There is no momentum or optimizer-state merge in v0.1.

## Development layout and next milestones

`model.py` defines `ModelConfig`, the shared model, and the CPU/MLX backends. `artifacts.py` hashes and assembles chunked checkpoints. `cluster.py` owns membership, reduction, and durable state. `worker.py` trains and retries requests. `observe.py` records events, inventories hardware, collects telemetry, and builds reports. `benchmark.py` probes public Accelerate BLAS.

The default trainer is still local SGD with token-weighted averaging of the 29,888-parameter model. The next milestones are a second physical Mac over Wi-Fi, the remaining MLX gradient and capture checks, a real corpus, calibrated batch selection, and an ANE adapter with independent gradient checks. Automatic memory admission, compression, LoRA, dynamic membership, and sharded large-model inference are not implemented. Artifacts larger than 64 MiB are rejected. Do not treat a wider model as finished scaling until memory and a real data split are measured.

The hardware inventory distinguishes M5 GPU neural accelerators from ANE and detects CPU SME/SME2 features where permitted. It also records media engines as a future preprocessing opportunity. Inferred presence never counts as verified execution, and unknown future chip generations are not automatically declared supported.
