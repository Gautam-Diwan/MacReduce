# Profiling and accelerator research

The primary question is where time and energy go in a completed training round. System utilization alone cannot answer it. Keep four forms of evidence separate: hardware specification, API availability, executed operations, and measured improvement to the workload.

## What the implementation records

| Layer | Implemented measurement | Interpretation |
|---|---|---|
| Training | Per-step completed-work duration, loss, tokens, selected backend | MLX explicitly evaluates its lazy graph before a step ends. CPU steps are synchronous. Includes forward, backward, and SGD update. |
| Distribution | Checkpoint import/export, HTTP request durations, successful request payload bytes, retries, barrier wait | HTTP durations include server work. Bytes exclude TCP/IP overhead and unknown delivery on failed attempts. |
| Reduction | Decode, weighted average, held-out evaluation, disk commit | Coordinator work can become a bottleneck. |
| Memory | Sampled process RSS; MLX active and peak allocator bytes at round end | RSS is sampled, MLX counters are process allocator values, neither is total device pressure. |
| Operator detail | One PyTorch CPU step with shapes and memory events; optional MLX Metal capture | Captures are opt-in and alter performance. Collect separate uninstrumented timings. |
| Hardware | Capability JSON, raw and normalized mactop samples | Power readings are system-wide estimates; unavailable samples are recorded as unavailable. |

`report` produces phase p50/p95 timings, local compute rates, validation loss per round, transfer totals, retry count, and per-node Perfetto/Chrome trace JSON. Nested spans overlap. Do not sum every table row as elapsed time. Startup, warmup, and profiling costs remain visible rather than being silently discarded. The first prototype reports compute rates separately from end-to-end throughput.

## Apple tools for actual placement

Use full Xcode and Instruments from a normal Mac session. `xcrun --find xctrace` checks whether the command-line Instruments tooling is installed; Command Line Tools alone did not provide it on the development host. Available instrument names and counters vary by Xcode, macOS, and chip, so inspect the installed templates before automating captures.

For Core ML or a future ANE adapter, correlate model calls with the **Core ML instrument and Neural Engine activity**, then add CPU and GPU tracks to identify surrounding work. Framework compute-unit preferences do not guarantee every operation ran on ANE. Private ANE execution may not produce the same high-level Core ML events, so add explicit application intervals and confirm low-level activity. Apple's [Core ML profiling session](https://developer.apple.com/videos/play/wwdc2022/10027/) explains the supported instrumentation.

For MLX, use **Metal System Trace** for scheduling and host/device interaction, and a **GPU capture** for kernel-level inspection. The implementation can capture the second local step, after one warmup step:

```sh
MTL_CAPTURE_ENABLED=1 macreduce worker \
  --url http://127.0.0.1:8765 --node m4 --backend mlx-gpu \
  --output runs/wifi/events --capture runs/m4-step.gputrace
```

The destination must not already exist, its parent must exist, and the job needs at least two local steps. Open the capture in Xcode. This path is implemented but could not execute in the sandbox. See [MLX's capture documentation](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html).

The CPU reference path can capture an operator trace:

```sh
macreduce worker --url http://127.0.0.1:8765 --node cpu \
  --backend torch-cpu --output runs/cpu/events \
  --torch-trace runs/cpu-operators.trace.json
```

Open `.trace.json` files in a Perfetto-compatible trace viewer. CPU events show operators, shapes, and memory, but do not identify SME or AMX instructions. The delivered package includes a real operator trace from a separate instrumented demo. See [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html).

## mactop and powermetrics

The optional mactop collector retains raw JSON, stderr, command, version, and normalized fields. Its known power fields are expressed in watts, temperature in Celsius, and GPU utilization in percent. Missing fields stay null. New or chip-specific channels remain in raw data instead of receiving guessed units. mactop uses Apple system interfaces and its reported metrics can include indirect estimates, particularly across chip generations; retain the raw provenance when comparing machines. [Collector documentation](https://github.com/metaspartan/mactop).

Apple's built-in `powermetrics` is an independent supplementary source. The local help on macOS 26.5 lists `cpu_power`, `gpu_power`, `ane_power`, and `thermal` samplers. Inspect `powermetrics --help` on each machine first. From an ordinary terminal, an example capture is:

```sh
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal \
  --sample-rate 1000 --sample-count 30 --format plist \
  --output-file runs/powermetrics.plist
```

This manual capture requires administrative access; the trainer never invokes sudo. The plist stream may contain NUL-separated records, and field units must be read from the captured schema. v0.1 does not ingest or normalize powermetrics files. The local tool warns that SoC power values are estimates and unsuitable for direct device-to-device comparisons. Record idle baseline and background workload, and use an external meter for stronger whole-system energy comparisons. No ANE/GPU power measurements were obtained in this sandbox.

## Hardware beyond ANE

| Resource | What to check | Current treatment |
|---|---|---|
| CPU NEON/SIMD | Architecture, operator timings, vectorized library calls | CPU reference executes; exact vector instruction dispatch is unverified. |
| CPU matrix operations | Accelerate BLAS; `hw.optional.arm.FEAT_SME` and `FEAT_SME2` | Public BLAS probe runs. Feature queries are preserved as unavailable when blocked. |
| GPU | MLX matmul smoke test, memory counters, Metal traces | Backend exists; execution blocked in this sandbox. |
| ANE | M-series specification, usable runtime, gradient correctness, actual activity | All M-series start with ANE at M1. Training adapter remains future work. |
| M5 GPU Neural Accelerators | M5 specification, supported tensor kernels, dispatch trace | Recorded separately from ANE; M4 reports this feature absent. No M5 hardware result is claimed. |
| Media engines | VideoToolbox decode/encode availability and input-pipeline timing | Relevant to future video datasets, not dense Transformer math. |
| Unified memory and thermals | Available RAM, swapping, DRAM bandwidth, thermal pressure, sustained clocks | Inventory and raw telemetry only; no automatic scheduling policy yet. |

Apple documents [Accelerate](https://developer.apple.com/documentation/accelerate) as public CPU numerical infrastructure. Apple’s [XNU SME documentation](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md) describes SME/SME2 state, while [Hello SME](https://arxiv.org/abs/2409.18779) measures matrix kernels on M4. Legacy Apple AMX and Arm SME should not be conflated with ANE or with Intel's similarly named AMX. Use supported APIs first and leave internal dispatch unverified until traced or otherwise demonstrated.

The `matmul` command calls Accelerate's `cblas_sgemm` directly with float32 inputs, three warmups, repeated synchronous timings, and sampled float64 dot-product checks. The development host executed the 512-by-512 probe correctly. Its calculated GFLOP/s is a short microbenchmark, not sustained training throughput and not proof of AMX/SME dispatch. Run longer workload-shaped experiments before drawing conclusions.

M5 has a Neural Accelerator within each GPU core, separate from the chip's Neural Engine. Apple explains access and debugging in [its M5 GPU technical talk](https://developer.apple.com/videos/play/tech-talks/111432/). A new tensor API being available on an older GPU does not imply that GPU has the M5 hardware. Presence on M5 also does not establish that a given MLX kernel uses it. The inventory marks dispatch as unverified; future generations are left unknown until checked.

## Wi-Fi and heterogeneous benchmarking protocol

1. Record hardware/runtime versions, power source, network topology, worker count, batch sizes, local steps, dataset fingerprints, and effective payload rates. Do not use advertised radio speed as effective throughput.
2. Run one Mac, then two or three, at matched token budgets and a fixed validation set. Compare K=1, 10, and 50, using enough rounds to expose convergence differences. Keep total processed tokens comparable.
3. Record barrier time per worker and separate communication, conversion, compute, and validation. Collect concurrency measurements because Wi-Fi airtime and memory bandwidth are shared.
4. Repeat trials and distinguish cold start, warmup, and sustained performance. Capture a short detailed trace separately, then quantify instrumentation overhead.
5. Exercise interrupted links and worker restarts. A retry must not duplicate an optimizer update. Deliberate membership changes require a new, documented training policy.
6. For any ANE adapter, first compare forward values, gradients, and one optimizer step against the reference. Then measure compilation, weight materialization, copies, host-side gradients/optimizer work, and complete step time. Accelerator busy time alone is insufficient.

Before enabling automatic batch assignment, define the intended data mixture. Giving a faster node more tokens changes its influence if nodes have different datasets. For LoRA, averaging A and B factors independently does not generally equal averaging the full weight deltas; an adapter reducer must specify and test its mathematical operation.
