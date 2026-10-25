# MacReduce: Distributed Training on Heterogeneous Macs

Project proposal • Updated October 23, 2026 • MacReduce

We propose an observable training framework for Apple silicon Macs connected over Wi-Fi. The central question is whether infrequent model synchronization and hardware-aware work assignment can improve time to a fixed validation loss when machines differ in memory, compute speed, and thermal limits. A working small-model prototype provides the starting point; measurements on multiple physical Macs will determine whether the approach scales.

## Research questions

- How many local steps should each worker perform between exchanges before reduced communication is outweighed by model drift?
- Can different batch sizes reduce straggler delays while preserving an explicitly defined training objective?
- When does an additional compute path, including ANE, CPU matrix operations, or M5 GPU neural accelerators, improve complete training steps or energy use after conversion and transfer overhead?

## Scope and expected contribution

The core deliverable is a trainer for small language models, a coordinator and worker protocol, and reproducible profiling across heterogeneous Macs. We will begin with local stochastic gradient descent and periodic token-weighted parameter averaging. This is an iterative map and reduce pattern: workers train from a common checkpoint, the coordinator reduces their updates, and the next round begins.

Every training worker initially holds the complete model. This design therefore does not pool memory for a model larger than an individual worker can hold. Training a 30B–100B model from scratch and sharding a large model for inference are separate engineering problems. Large-model inference remains a stretch experiment using an existing inference engine, after the trainer is evaluated.

The contribution is a measured study of hardware use, communication, and heterogeneous scheduling on Wi-Fi. Distributed training on Macs already exists in MLX and AirTrain; novelty should not be claimed for that capability alone. DiLoCo supplies a useful low-communication comparison, but its inner AdamW and outer Nesterov optimizers differ from our initial local-SGD implementation. [1–3]

## Architecture and WiFi operation

A coordinator owns the current model version, fixed worker membership, validation set, and committed checkpoints. Workers register their selected backend, memory and hardware inventory, batch size, and dataset fingerprint. Each round distributes the same checkpoint, performs K local steps per worker, and gathers one update from every admitted worker.

The reducer averages parameters using each worker’s processed token count. With one local step and equal learning rates, this matches pooled-batch SGD, within numerical tolerance. With multiple local steps, parameter averaging is a different optimization trajectory; we will compare validation curves rather than assume equivalence. Unequal worker batch sizes also change the contribution of their local data, so the target data mixture must be specified.

## Transport and recovery

The first implementation uses HTTP over TCP with a shared bearer token, optional TLS, checksummed NumPy checkpoints without pickle, run identifiers, and monotonically increasing versions. Retransmitting the same update is idempotent. Conflicting retries, future updates, and stale updates without an existing receipt are rejected. The coordinator commits the checkpoint before acknowledging a completed round.

A round deadline reports missing work and fails the run instead of silently removing a participant. Restart restores the last committed version; an incomplete round is replayed from that checkpoint. Mini-batches are seeded by worker rank and round. Automatic quorum changes, elastic membership, chunked transfers, and concurrent coordinator replicas are later work.

## Bandwidth budget

Let S be the serialized model size and B the measured application payload rate. A worker needs approximately 2S/B seconds to download and upload one round, before latency, retries, and contention. For a 30M-parameter float32 model, S is about 120 MB. At an assumed 50 MB/s payload rate, this is at least 4.8 seconds per worker per round. These are planning values, not measured Wi-Fi results.

Clients sharing an access point compete for airtime; client-to-client transfers may traverse the radio twice. We will measure aggregate throughput under concurrent workers rather than substitute the advertised Wi-Fi link rate. Increasing K amortizes transfer cost, but may worsen convergence. Adapter updates and compression are follow-up options if communication dominates.

## Heterogeneous workers

The initial version supports explicit per-worker backend selection and batch sizes with one common K. We will calibrate step time and memory headroom before extending this to automatic batch selection. GPU name and advertised TOPS are insufficient scheduling inputs. Sustained performance, available memory, swap, temperature, and competing workloads all matter. Mixed numerical backends require forward and update parity checks before experimental results are compared.

## Hardware inventory and profiling

All M-series Macs, starting with M1, include an Apple Neural Engine. The M5 also adds a Neural Accelerator to each GPU core; these are distinct from the ANE. CPU matrix operations, including SME support introduced with M4, provide another route worth benchmarking. Hardware presence, runtime availability, and verified operation placement will be recorded separately. [4–7]

| Compute path | Access and evidence |
| --- | --- |
| CPU SIMD and matrix operations | Public Accelerate BLAS probe; feature queries for SME and SME2. A BLAS result does not identify the internal unit that executed it. |
| GPU through MLX | Training backend and explicit GPU smoke test. Use completed-work timing, MLX allocator counters, Metal capture, and Metal System Trace. |
| Apple Neural Engine | Experimental training adapter after a correctness gate. Use ANE activity traces and end-to-end timings, including CPU work and copies. |
| M5 GPU Neural Accelerators | Record hardware capability and verify kernel dispatch on an M5. M4 cannot validate this path; no speedup is assumed. |
| Media engines and shared memory | Video decode could assist future multimodal input pipelines. Memory bandwidth, pressure, swap, and thermal limits constrain concurrent accelerators. |

## Measurement hierarchy

Application events identify run, worker, model version, step, backend, processed tokens, duration, and payload bytes. Separate phases cover batch preparation, training, checkpoint conversion, network requests, waiting, reduction, validation, and disk commit. Per-node timeline files use local monotonic clocks; unsynchronized clocks will not be presented as one accurate cluster timeline.

Hardware attribution uses Apple Instruments. Core ML and Neural Engine instruments distinguish framework activity from ANE execution; Metal System Trace and GPU captures examine GPU scheduling and kernels. The MLX capture interface can save a selected step for inspection in Xcode. For the CPU reference backend, a PyTorch operator trace identifies costly operators and their shapes. [8–10]

mactop and powermetrics provide supplementary system telemetry. Power readings are estimates, and background activity can contribute. They do not prove that this trainer used the ANE. Missing counters remain unavailable rather than becoming zero. Record collector version and raw samples because available channels and estimation methods vary by chip and OS. [11]

## Evaluation and acceptance criteria

The minimum experiment uses two physical Macs on the same Wi-Fi network, including the available M4. A third machine adds a stronger heterogeneity test. We will record model, OS, runtime versions, RAM, power source, network setup, effective transfer rate, and background load. M5 results are optional and will remain explicitly untested without hardware.

| Comparison | Purpose |
| --- | --- |
| One Mac and K equal to 1 | Establish a training and synchronization baseline using matched data and token budgets. |
| Two or three Macs with K in 1 10 50 | Measure the communication versus convergence tradeoff at equal processed tokens and time to a fixed loss. |
| Equal versus unequal batch sizes | Test straggler reduction and document the resulting data weighting. Keep K fixed for this comparison. |
| Normal Wi-Fi and injected delay or interruption | Measure retries, deadline behavior, lost-response handling, and recovery without changing membership. |
| Accelerator isolation and overlap | Compare CPU, GPU, and experimental ANE paths first in isolation, then concurrently, including shared-memory contention. |

## Correctness and quality

A one-step numerical test must match the pooled-gradient reference. Checkpoint parsing must reject incorrect shapes, dtypes, non-finite parameters, and invalid archives. Duplicate submissions must not count twice. A restart must preserve the committed model and receipts. Causal attention must not expose future tokens. These checks are in the initial test suite; MLX parity is an explicit hardware-dependent test.

The current synthetic byte-language task checks the plumbing. It is not evidence of useful language-model quality. The next workload should be a licensed text corpus with a documented, disjoint train and validation split, a fixed tokenizer, and a manageable 1M–30M-parameter model. Dataset fingerprints and seeds must be retained. Parameter-efficient fine-tuning can follow once the transport and training objective are stable.

## Performance and profiling

Report time to a fixed held-out loss, validation loss versus processed tokens, useful tokens per second, per-worker step-time distributions, barrier wait, bytes per token, transfer time, retries, and memory pressure. Separate startup and warmup from steady state. Repeat timed trials at least three times and show variability. Measure profiling overhead with collectors enabled and disabled.

The project succeeds if it produces a correct recoverable trainer and explains the observed tradeoffs with reproducible evidence. Positive speedup is a hypothesis, not a promised result. A proposed accelerator should be retained only after matching numerical behavior and demonstrating an end-to-end benefit for an identified workload.

## Implementation status and delivery plan

Version 0.1 now contains a causal byte Transformer, a CPU reference trainer, an MLX backend, a real HTTP coordinator and worker loop, token-weighted reduction, checkpoint recovery, JSON event logs, per-node timelines, a Prometheus status endpoint, capability probes, optional mactop collection, and an Accelerate matrix benchmark.

The initial local test ran two CPU worker processes with batch sizes 2 and 4 for eight rounds of ten steps. Fixed synthetic validation loss decreased from 6.0587 to 1.7681. This demonstrates learning through the distributed protocol on loopback; it does not demonstrate physical multi-Mac scaling or Wi-Fi performance. A direct Accelerate matrix operation also passed a sampled numerical check.

Metal initialization and mactop hardware queries are blocked in the current execution sandbox. The MLX backend and GPU capture path are implemented but require validation from a normal Mac terminal. Full Xcode profiling tools are not installed in this environment. ANE training, automatic work assignment, adapter training, and sharded large-model inference are not implemented in version 0.1.

| Target date | Deliverable |
| --- | --- |
| October 22 to November 5 | Validate MLX outside the sandbox, run two Macs over Wi-Fi, capture a baseline, and finalize the proposal scope. |
| November 6 | Submit the proposal by 11:59 PM ET, if using the supplied course schedule. Confirm course fit and team responsibilities. |
| November 7 to November 15 | Evaluate K and batch-size choices on a real corpus; test restart and interrupted-network behavior. |
| November 16 to November 24 | Add calibration-based batch assignment. Evaluate an ANE adapter only if local correctness and profiling gates pass. |
| November 25 to December 4 | Repeat experiments, quantify profiling overhead, document limitations, and prepare the final report by 11:59 PM ET. |

## Risks and scope decisions

Wi-Fi may dominate runtime, especially as models grow. ANE training relies on experimental software paths and may not beat GPU execution after overhead. All accelerators share memory bandwidth and power limits, so their standalone performance cannot simply be added. A slow worker can dominate a fixed barrier. These risks are subjects of measurement and reasons to keep the baseline small.

If ANE work stalls, the deliverable remains a heterogeneous Mac training study with validated profiling. Large-model inference is a separate stretch goal: use layer sharding in an existing engine and measure prefill and decode latency separately. It must not displace the trainer evaluation. [12–13]

## Refined research direction and implementation guide

The recommended first study is objective-preserving scheduling under changing Wi-Fi and thermal conditions. Define dataset mixture weights independently of hardware speed, establish equal effective batches with device-specific micro-batches and accumulation, then compare static calibration with a bounded adaptive policy. Measure calibration cost, uncertainty, held-out quality, and complete runtime. Unequal local horizons need an explicitly validated aggregation rule; changing token weights is not an automatic solution.

This remains a research hypothesis, not a verified novelty claim. AdaComm, FedNova, HEW-LocalSGD, AirTrain, and FusionML overlap with important parts of the idea. The [related-work map](../wiki/research/related-work.md) states reading scope, and [research candidates](../wiki/research/novelty.md) define baselines, ablations, and falsification criteria. Full-paper review of the closest methods is required before making a novelty claim.

The [implementation roadmap](../wiki/project/roadmap.md) links detailed guides for MLX validation, physical Wi-Fi, model and protocol scaling, heterogeneity, ANE integration, profiling, M5 and CPU acceleration, and sharded inference. The [evaluation plan](../wiki/research/experiments.md) specifies staged acceptance gates. The maintained [project state](../wiki/project/state.md) separates measured results from future work.

## Selected references

Primary documentation and original research, checked October 22, 2026. Numbered references support the design choices above. The accompanying feasibility report contains the wider background review.

[1 MLX distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)

[2 AirTrain distributed training on Apple silicon](https://github.com/alexandercodes4/AirTrain)

[3 DiLoCo Distributed Low Communication Training of Language Models](https://arxiv.org/abs/2311.08105)

[4 Apple introduces M1 with a 16 core Neural Engine](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/)

[5 Apple M5 GPU Neural Accelerators](https://developer.apple.com/videos/play/tech-talks/111432/)

[6 Hello SME matrix multiplication on Apple M4](https://arxiv.org/abs/2409.18779)

[7 Apple XNU documentation for SME and SME2](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md)

[8 Apple Optimize your Core ML usage](https://developer.apple.com/videos/play/wwdc2022/10027/)

[9 MLX Metal debugger and GPU capture](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)

[10 PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)

[11 mactop collector documentation and source](https://github.com/metaspartan/mactop)

[12 Experimental ANE training implementation](https://github.com/maderix/ANE/tree/main/training)

[13 EXO distributed inference](https://github.com/exo-explore/exo)

[14 Apple Accelerate framework](https://developer.apple.com/documentation/accelerate)

Local evidence is delivered with the source package under evidence. It contains run settings, observed results, capability failures, training traces, and the test summary. The current prototype uses standard local SGD; the DiLoCo paper’s performance results are not attributed to this implementation.
