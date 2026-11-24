---
title: "Three-Mac wireless federated QLoRA: recipe and networking; AWS background"
type: research
status: proposed
updated: 2026-11-23
sources: ["S162", "S163", "S164", "S165", "S166", "S167", "S168", "S169", "S170", "S171", "S172", "S173", "S174", "S175", "S176", "S177", "S178", "S179", "S180"]
---

# Three-Mac wireless federated QLoRA: recipe and networking; AWS background

**Implementation follow-up:** the [Tailscale + Flower runbook](../guides/tailscale-federation.md) now supplies a remote pilot path with persistent local trainers. The code-gap audit below describes the older `federate --execute` path. The user deferred batch/sync/data-allocation calibration; it is [pinned as TODO](../project/roadmap.md). Standalone ANE remains gated.

## Recommendation and inspected scope

**Active constraint, confirmed 2026-11-12:** use only the team's three Macs. There is no separate AWS coordinator, rented worker, or fourth machine in the study. The AWS research below is background only.

**Proposed:** start with GPU-only QLoRA on three full replicas, an uncompressed fp32 adapter payload, a common optimizer-update horizon, and per-device microbatch/accumulation settings calibrated to balance round duration. The M4 Air 24 GB runs both its training client and a lightweight coordinator process; the M4 Pro 16 GB and M1 Air 8 GB run training clients. Use a shared reachable Wi-Fi LAN plus SSH forwarding first. If the Macs are on separate networks, use Tailscale private addresses to reach the same M4 Air, without an additional application server. Compare two M4s with all three Macs. Treat GPU+ANE and separate ANE clients as subsequent feasibility-gated arms.

**Hardware reported by the team on November 12:** M4 MacBook Air 24 GB, M4 MacBook Pro 16 GB, M1 MacBook Air 8 GB. The team can work on shared or separate networks; SSH is acceptable. These are user-reported inventories, not three locally inspected `doctor` outputs. The exact M1 GPU variant, OS versions, available RAM, sustained training speed, uplinks, and access-point behavior remain unknown. Apple lists Wi-Fi 6 and a 16-core Neural Engine for the M1 Air; neither is a training-throughput guarantee. [S179](../sources/S179.md)

**Inspected locally:** the [proposal](../../docs/proposal.md), [current config](../../configs/qlora_m4.yaml), [saved successful M4 curve](../../evidence/qwen3-0.6b-curve/2026-10-29T00-07-18/result.json), `federated.py`, `lora_client.py`, `artifacts.py`, and the heterogeneity, Wi-Fi, transport and optimizer guides. Read the M4 16 GB report from remote-tracking branch `origin/edmund/m4-16gb-baseline` at `a80a4ac`; its measurements are collaborator-reported here, not rerun. Inspected 19 newly saved primary sources with scope in [S162–S180](../sources/S162.md). No wireless experiment, M1 training run, cloud instance, or new ANE validation was performed. Numerical examples below are calculated planning scenarios, not measurements. No novelty claim is made.

## 1. What needs to be comparable

The proposal says batch 8 with accumulation 2. The successful saved curve actually used batch 1 and accumulation 1, rank 8, scale 2, sequence cap 512, 12,288 updates, AdamW peak 1e-4, weight decay 0.01, norm clip 1, 300-step warmup from 1e-7 and cosine decay to 1e-6. Its base revision is `73e3e38d981303bc594367cd910ea6eb48349da8`; WikiText revision is `b08601e04326c79dfdd32d625aee71d232d685c3`. Preserve that historical result and explicitly reconcile the proposal before new runs.

Use two controls, because they answer different questions:

1. **Historical anchor:** the existing batch-1 curve establishes continuity with the delivered single-Mac experiment. Reproduce GPU-only on the fastest M4 with the pinned recipe when possible. Its optimization trajectory need not match a multi-client run.
2. **Matched centralized control:** run the same logical global minibatch as the distributed per-step gradient baseline on one M4. Use the sum of the selected effective local batches for this control. For example, effective batches 1/2/4 require a centralized batch-7 control, implemented with accumulation if needed. Batch 3 applies only to the uniform batch-1 control. Match token-loss weighting, sample order, optimizer state, clipping and learning-rate progress. This is a new control, not a relabeling of the historical batch-1 run.

**Data budget:** partition the same 4,096 training examples once, deterministically and without overlap; expose each exactly three times in the primary fixed-work comparison. Do not give each client 12,288 updates over its own shard: that is roughly triple the intended work. For the uniform control, equal-count shards have 1,366 / 1,365 / 1,365 examples. For the hardware-aware arm, deterministically allocate random shards in proportion to selected effective local batches so each client needs approximately the same number of updates per epoch. Freeze the assignment before the K sweep, retain the same complete union of 4,096 examples, and derive aggregation weights from the chosen data objective, not directly from device speed. Handle final partial blocks explicitly; do not silently drop, duplicate, or repeatedly sample the first local batches. Keep per-client data cursors, epoch counters and RNG state across communication calls.

WikiText lines have unequal lengths. Record actual supervised non-padding tokens after truncation/tokenization; 12,288 example presentations are not 12,288 fixed 512-token sequences. Keep validation/test full and unchanged. Compute perplexity as exp(total negative log-likelihood / total supervised tokens), not a mean of batch perplexities. Select checkpoints/hyperparameters on validation, and reserve test for the final selected models.

**Objective:** define `F = sum_i p_i F_i` before optimizing hardware use. To reproduce a token-average language-model loss, use local token-normalized objectives and shard weights proportional to each shard's supervised-token mass; the per-step gradient control weights actual minibatch tokens. If retaining the historical trainer's precise example/batch weighting instead, preserve and disclose it across controls. Equal example counts do not prove equal token mass. FedAvg's original algorithm weights clients by their data objective, not RAM or speed. [S162](../sources/S162.md)

**Recipe proposed for the first federation:** same base/tokenizer/quantization hashes, rank 8, scale 2, dropout 0, target modules and layer count; sequence cap 512; fp32 communication; calibrated per-device effective batches; common K; AdamW and clipping from the saved recipe. Preflight microbatch 1 on the M1; use checkpointing there if needed. Activation checkpointing is a memory/computation choice to validate numerically, whereas reducing rank, layers or sequence length changes the study. MLX-LM documents accumulation and memory controls, but the exact M1 workload remains unverified. [S175](../sources/S175.md)

### Revised hardware-aware pilot configuration

**Revision after the user's idle-time objection:** the earlier recommendation of microbatch 1 / accumulation 1 / K=32 on every Mac is retained only as a uniform-work control. It should not be the main heterogeneous-throughput design. Equal work behind a barrier can leave the M4s waiting; different hardware does not require identical microbatches or effective local batches.

**Proposed procedure:** first measure warmed, length-matched batches on each actual device. Start the M1 at microbatch 1 and accumulation 1. Profile M4 microbatches 1,2,4 (larger only if memory permits), initially with accumulation 1, then accumulation 2 or 4 when needed. Record effective-update p50/p95 time, useful supervised tokens/s, padding, memory pressure and swap. Profile the M4 Air while its coordinator is active. Select efficient microbatches first; accumulation can give a faster device more useful examples per optimizer update when another large microbatch is impractical. Do not manufacture delay just to match the M1.

Let `b_i = microbatch_i * accumulation_i` and `t_i` be measured time for one such effective optimizer update. Select per-device settings that keep `t_i` reasonably close while maximizing aggregate useful tokens/s within memory/quality constraints. Use a common K initially, so predicted local completion is `K*t_i`. Balance these measured times, not nominal chip names or RAM sizes. No setting can guarantee zero waiting under thermal/network variability.

**Illustrative candidate, not measured settings:** M1 microbatch 1, M4 Air microbatch 2, M4 Pro microbatch 4, all accumulation 1. At K=32 this processes 32/64/128 examples, or 224 total per full round. It is not one batch-224 gradient; each client makes 32 sequential AdamW updates. If timings are still imbalanced, change each M4's efficient microbatch/accumulation combination. The candidate 1:2:4 ratio must not be inferred from hardware specifications or treated as established throughput.

Because WikiText is a shared public corpus in this study, assign larger randomly sampled shards to the devices taking larger effective batches, with no duplicated/dropped examples and exactly three total exposures per example. This keeps the global corpus while changing its partition. For a token-average objective, use intended shard supervised-token weights; for the synchronous-gradient control, reduce by actual minibatch token counts. If data were private and tied to owners, arbitrary repartition would not be available. With equal K and correctly normalized gradients, unequal batches have a straightforward pooled-SGD K=1 reference, but local AdamW/FedAvg at K>1 still requires empirical quality validation.

Keep AdamW peak 1e-4, weight decay 0.01, norm clip 1.0 and cosine end 1e-6 as starting recipe parameters. Map the warmup/decay to aggregate data exposure; with larger total batches, the old 100-tick or 300-tick warmup numbers cannot simply be copied. Persist local moments/counters and sampler state. Evaluate at each global epoch and checkpoint aggregate plus client recovery state on commit. Use a matched centralized batch of `sum_i b_i` in the true gradient control; keep the historical batch-1 curve separately.

K=32 remains an initial pilot horizon, not a fixed wall-clock interval or an established optimum. Sweep K=8,32,128 after calibration, holding the selected device batches/partition fixed across that sweep. Compare the uniform control and calibrated policy under the same total supervised-token exposure and time-to-quality, including barrier waiting. A provisional 30–60 seconds of local compute per round can be tested by converting that target to one common K after timing; it is a planning range, not a measured recommendation. Allow only declared final partial batches/rounds at epoch boundaries.

**Earlier control arithmetic:** the old batch-1 partition gives 43 rounds/epoch and 129 over three epochs at K=32. These counts and the old 96-examples/round figure apply only to the uniform control. For selected batches `b_i`, the balanced planning count becomes approximately `12288 / (K * sum_i b_i)`; exact epoch-boundary counts depend on frozen shards and partial batches. Recalculate rather than retaining the old sync count.

### Schedules and optimizer state are part of the experiment

Index the learning-rate schedule by common data progress, not communication round or wall time. Preserve the historical warmup fraction `300/12288` and warmup token exposure using the recorded token stream when available. Under the equal-length example approximation, the warmup is roughly `300 / sum_i b_i` common optimizer-update ticks. The old 100-tick number is specific to the uniform three-client batch-1 control. Actual token exposure and final partial batches require explicit schedule mapping. Each round carries the starting schedule position; clients advance it according to a predeclared common work plan. Report exact schedule values and actual token exposure rather than claiming token-exact matching from line counts alone.

**Primary policy proposed:** preserve each client's own AdamW moments and update counter across broadcasts; replace its adapter with the aggregate, but do not average moments. Checkpoint them locally with the global round and data cursor. This is a stated local-AdamW federated algorithm, not centralized AdamW. A secondary reset-per-round arm can quantify the effect, with a matched one-client segmented control. Resetting moments every round makes K an optimizer-reset intervention as well as a communication intervention; do not hide that confound.

For the true per-step all-reduce control, aggregate gradients at common adapter parameters and apply one common AdamW update, with shared optimizer state and clipping after token-weighted reduction. Average-of-local-AdamW-updates is generally different from AdamW-of-average-gradient. The repository's `algorithm="allreduce"` currently averages locally updated adapters at K=1; its label alone does not establish the proposal's synchronous-gradient control.

Keep equal K initially. FedNova studies objective inconsistency under unequal local work; the current scalar delta/K correction is the plain-SGD special case, not an established correction for adaptive AdamW moments. If unequal K is explored, retain a plain-SGD numerical experiment or implement a justified solver-specific algorithm as its own arm. [S163](../sources/S163.md)

### LoRA aggregation: choose and name the baseline

Factor FedAvg is a practical, fixed-rank baseline: average corresponding adapter parameters from a common initialization and document that choice. It is FedAvg in adapter parameter space. It is not the exact mean of the represented dense weight updates:

`sum_i p_i (B_i A_i) != (sum_i p_i B_i)(sum_i p_i A_i)` in general.

This distinction does not make factor FedAvg unusable. It limits what mathematical equivalence can be claimed. Keep both factors trainable to preserve the existing single-Mac model class, and log this as an approximation relative to dense-update averaging. A common frozen factor would make aggregation linear, but requires fresh matching single-Mac baselines.

FLoRA stacks factors to represent the product average and supports heterogeneous ranks; its broadcast representation is larger. FedEx-LoRA instead incorporates a residual into frozen weights and describes low-rank residual communication. Both require explicit integration decisions for a fixed, quantized base and client resume state. Neither should be substituted casually into the first network experiment. Read sections 2–3 of FLoRA and 3–4 of FedEx-LoRA; no full-paper novelty review or Mac reproduction was performed. [S164](../sources/S164.md), [S165](../sources/S165.md)

## 2. Connecting without inter-machine cables

Wi-Fi is a link technology. TCP is a transport; HTTP is an application protocol; TLS, SSH and WireGuard provide different encrypted paths. SSH can carry the existing HTTP protocol without exposing its port publicly. Changing to SSH does not remove the radio/uplink bandwidth limit.

**Shared network:** use an existing private 5 GHz Wi-Fi LAN or access point with client-to-client reachability. No new router purchase is required to begin. University/guest isolation may block peers despite a shared SSID. The coordinator is the M4 Air 24 GB. Measure reduction/evaluation contention there and use the same coordinator throughout comparable arms. A co-located coordinator is also one training client.

**Different networks:** Tailscale gives stable private addressing and encrypted connectivity. Its retrieved Personal plan lists six users at no charge, enough for this three-person project if its usage terms fit. Determine the actual path with `tailscale status`, `tailscale ping <peer>` and `tailscale netcheck`; capture path changes during runs. Direct, peer-relayed and DERP trials must not be pooled into one performance number. Direct connections usually have better throughput/latency; strict NAT or blocked UDP can force relaying. [S166](../sources/S166.md), [S167](../sources/S167.md)

**SSH-only application access:** once the coordinator's SSH endpoint is reachable, bind its application to `127.0.0.1:8765` and establish this on each remote Mac:

```sh
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18765:127.0.0.1:8765 \
  USER@REACHABLE_COORDINATOR
```

Workers use `http://127.0.0.1:18765` through that tunnel, retaining application authentication. This is a connection template; it does not make the current multi-Mac QLoRA path runnable. On the coordinator itself, use its local application port. Remote SSH still needs reachability: use the M4 Air's Tailscale name or reachable LAN address. SSH alone does not traverse two arbitrary home NATs. [S180](../sources/S180.md)

For a plain file-based prototype, persistent SFTP/SSH sessions can exchange versioned immutable adapter files and receipts; upload to a temporary name, verify hash and atomically publish completion. Avoid launching a new `scp` per chunk and avoid syncing an entire run directory. SSH transport does not implement round identity, crash recovery or exactly-once reduction for you. Tunneling existing bounded HTTP artifact exchange is likely less project-specific work than replacing the protocol; this is an engineering judgment.

**Coordinator ownership:** the coordinator is a CPU-side process on the M4 Air 24 GB, alongside its GPU training client. It collects the two remote adapter deltas, includes the Air's local delta, commits the weighted aggregate, and returns it to the Pro and the M1. Evaluate the global adapter on the Air at the same prescribed intervals in each arm, pausing local GPU work when necessary and including that time in end-to-end measurements. Measure host-memory/CPU contention rather than assuming coordination is free. Keep all three Macs awake and plugged in. If the Air sleeps or disconnects, pause and recover from the last committed checkpoint; an AWS fallback or automatic coordinator election is not part of this first implementation. Tailscale may use its existing network relays when direct connectivity fails; that does not require provisioning another coordinator, and the path must still be measured. [S166](../sources/S166.md)

**MLX ring alternative:** official MLX documentation describes TCP ring communication and SSH/hostfile launches, which can operate over reachable IP links. Explicitly choose the backend and verify world size; default `any` can initialize a singleton. This is a useful GPU synchronous-collective baseline, with different failure semantics. JACCL's Thunderbolt RDMA path does not meet this study's cable-free constraint. [S176](../sources/S176.md)

**Framework alternative:** Flower supplies a SuperLink/SuperNode deployment runtime and TLS. It can replace orchestration/transport work, but still needs a custom persistent MLX QLoRA client, the chosen aggregation semantics and correctly counted tokens. Pin the chosen version; the current docs inspected were v1.43.x. Test whether client execution retains a resident model rather than reloading it every round. Switching frameworks is optional, not evidence that the current implementation is already distributed. [S177](../sources/S177.md), [S178](../sources/S178.md)

## 3. Quantifying communication and over-syncing

**Locally recorded input:** the successful saved curve lists 5,046,272 trainable adapter parameters. **Calculated:** a raw fp32 upload is `S = 4P = 20,185,088 bytes` (20.185 MB or 19.25 MiB), before serialization/protocol overhead. Upload plus next broadcast is about 40.37 MB per remote client per round. No training records, base weights, optimizer moments, telemetry or plots need to move in the critical round path. Cache pinned base weights once on each client and verify local data hashes.

For the active star on the M4 Air, two remote clients (Pro and M1) give approximately `4S = 80.74 MB/round` of aggregate application traffic. The Air's own update is local and does not traverse the network. At 128 idealized K=32 rounds, aggregate adapter traffic is about 10.33 GB; at K=128 and 32 idealized rounds, about 2.58 GB. These counts are adapter payload traffic, not radio airtime or full packet captures. Same-radio client-to-client relaying can consume additional airtime.

A conservative planning model is:

`round_s ~= max_i(K_i * warmed_step_s_i) + communication_s + reduction_s + checkpoint_s`

For separate remote links with concurrent transfers, model upload and download phases separately using the slowest client and M4 Air aggregate-link bottlenecks. For a shared AP, use measured aggregate goodput `B_agg`, so `communication_s ~= aggregate_payload_bytes / B_agg + protocol_latency`. Do not assign every simultaneous client its isolated bandwidth.

With the local-coordinator fp32 payload above and **assumed** aggregate goodput 10 MB/s, the transfer-only cost is 8.07 s/round. At an **assumed** slowest local step of 0.6 s, communication shares are about 93% at K=1, 63% at K=8, 30% at K=32, and 9.5% at K=128, before other costs. For a target share epsilon, `K >= ((1-epsilon)/epsilon) * communication_s / slowest_step_s`; this finds an amortization threshold, not a quality-optimal K.

At 100 / 20 / 5 Mbps effective per-client goodput, a 40.37 MB upload-plus-download has a transfer-only floor of about 3.23 / 16.15 / 64.59 seconds. These are illustrative rates, not measurements or simultaneous-LAN predictions. Mbps are bits/s; MB/s are bytes/s.

**Protocol latency can dominate:** 20,185,088 bytes is 77 chunks of 256 KiB before archive overhead. If every upload/download chunk is a serialized request-response with 40 ms RTT, 154 chunk acknowledgements add roughly 6.16 s to one client's round, before control messages. Archive metadata may add another chunk. Measure request count and pipelining, not just link Mbps. Keep bounded chunks but consider a small bounded window and connection reuse rather than serial stop-and-wait. The current implementation must be inspected before assuming overlap.

**Uniform-work control only:** equal batches and equal K give a simplified speedup ceiling `n * fastest_single_step_s / slowest_client_step_s`, ignoring networking, evaluation, thermal changes and optimization differences. Three clients only beat the fastest single Mac in this simplified fixed-work model if the slowest step takes less than three times the fastest step. Adding the M1 to two equal-speed M4s only helps compute throughput if its step takes less than 1.5 times an M4 step. Different M4 speeds require the full measured model. Longer K amortizes networking but does not fix this barrier imbalance. The revised main arm instead calibrates different effective local batches and allocates data accordingly; use its actual per-device times and examples/tokens for the throughput model.

**Proposed sweep:** K in {1, 8, 32, 128}; add 512 only if WAN cost makes smaller values impractical. Begin short preflights with K=32 after device batch calibration, then choose full-run candidates from measured end-to-end round times. For the uniform control only, a budget of 12,288 example presentations over three batch-1 clients gives an idealized `4096/K` rounds: 4096, 512, 128, 32. Exact counts can increase at shard/epoch/evaluation boundaries; log all partial rounds. Compare quality versus supervised tokens, time-to-predeclared-validation-target, total bytes, transfer p50/p95, compute time and barrier waiting.

K is too small when communication dominates; it is too large when local drift, slower convergence, or replayed work outweigh reduced traffic. Do not pick K using throughput alone. Hold all optimizer-state, participation and compression policies fixed while sweeping it. A one-shot merge at the end is a useful separately named endpoint, not automatically an adequate substitute for iterative federation.

## 4. Reliability design and measurements

**Primary scientific policy proposed:** a fixed three-client cohort commits only after all expected valid updates arrive. If a deadline expires, leave the round uncommitted and resume/replay it. A two-of-three quorum changes participation and often the objective; evaluate it as a separate arm with inclusion counts and weighting, not a transparent reliability tweak.

Carry run ID, global version, client ID, base/tokenizer/data/adapter-schema hashes, recipe and round-plan hashes, and explicit objective weights. Reject stale versions, NaNs, wrong shapes/dtypes, different models and conflicting retries. Persist an update's exact outgoing bytes and receipt until commit. A duplicate upload after an acknowledgement is lost must not count twice. Use temporary files plus validated completion and atomic checkpoint publication; distinguish process-restart guarantees from power-loss durability.

For restartable local AdamW, checkpoint global adapter/version plus each client's optimizer moments/update counter, RNG, sampler/data cursor and any quantization residual. The primary persistent-moment policy requires coordinated recovery of those states. A recovered client should not silently use fresh moments or replay a different batch stream. SSH keepalives detect a broken tunnel; a supervisor may reconnect it, but correctness comes from application versioning and durable receipts. Keep secrets/private data out of manifests and Git.

**Preflight and fault-injection sequence proposed:**

1. Inventory all machines; synchronize dependency lock/source and model/data hashes. On the M1, do a finite forward/backward, at least 200 warmed training steps and a longer thermal/memory check. Record process RSS, MLX memory, memory pressure and swap. Report unavailable sensors as unknown. Checkpointing is permitted, silent backend fallback is not.
2. Transfer adapter-shaped payloads (about 20 MB fp32, also test 1/5/40 MB) in both directions, alone and with both other clients active. Repeat under training load on LAN and remote paths. Capture bytes, serialization/hashing, actual goodput, RTT, chunk requests, p95 latency and path type. A nominal Wi-Fi rate or a tiny toy checkpoint is insufficient.
3. Interrupt a transfer midway; drop the acknowledgement after server acceptance; retry duplicates/conflicts; disconnect a client; restart the coordinator and a worker. Compare committed version/update identity and resumed state against the fault-free reference.
4. Predeclare round deadlines from observed p95 local compute plus p95 transfer and a stated margin. Record failed attempts/replayed work. Avoid an aggressive timeout that systematically excludes the M1.
5. Run plugged in, prevent sleep during active training, document background workloads and power mode, and warm fanless machines before sustained-speed claims. Local clocks may not agree; use per-phase monotonic durations and round IDs rather than subtracting wall timestamps across Macs.

## 5. AWS background only: excluded from the active three-Mac study

The user ruled out a separate AWS coordinator and asked to assume only the three owned Macs. This section preserves the originally requested cost research; none of its cloud architectures or additional machines is required or recommended for the active plan.

**Verified source prices below are USD, checked November 12, 2026; taxes, credits and negotiated discounts excluded.** Mac host prices come from AWS's November 10 feed for US East (N. Virginia), not a third-party calculator. The [saved extract](../../raw/sources/S170.json) retains source metadata, rate codes and response hash. AWS availability, quotas and ANE/private-API behavior remain untested. [S170](../sources/S170.md)

**A. External coordinator/relay — excluded by the current constraint.** A Linux Lightsail IPv4 bundle is listed at $7/month for 1 GB, or $12/month for 2 GB, with bundled disk/transfer. Prefer the 2 GB option as an initial headroom estimate for Python plus adapter reduction; actual peak RSS determines sizing. It does not run MLX/ANE training. A small EC2 Linux instance is another option; this research did not verify a live t4g hourly quote. Do not add a GPU, managed NAT gateway or load balancer solely to average three adapters. This hub improves reachability/availability, but adds WAN traffic and cannot accelerate an M1's training. [S171](../sources/S171.md)

For 128 idealized K=32 rounds and three remote fp32 clients, external-hub egress is `128 * 3S ~= 7.75 GB`, plus similar ingress. At K=1, egress is about 248.03 GB. EC2's shared 100 GB/month Internet-egress allowance may cover the first case if otherwise unused; it is shared with other usage, not per run. Beyond it, use a current region quote; $0.09/GB is an illustrative rate in the AWS VPC example, not a universal promise. Lightsail counts both ingress and egress toward its bundle; regional allowances differ. [S172](../sources/S172.md), [S173](../sources/S173.md)

EC2 public IPv4 alone is $0.005/address-hour, or $3.65 at 730 hours. The inspected NAT example adds $0.045/hour per AZ plus processing; a single continuous month at that hourly rate is $32.85 before bytes. EBS and logs are additional; the EBS page's $0.08/GB-month example is a planning assumption unless the chosen region is quoted. Leave these out of the model only when the selected product bundles them. [S173](../sources/S173.md), [S174](../sources/S174.md)

**B. Rent Apple Silicon Mac workers — useful controlled extension, poor substitute for this hardware study.** AWS lists M1 `mac2.metal` at $0.65/host-hour and M4 `mac-m4.metal` at $1.23/host-hour in the retrieved region. With the 24-hour minimum, compute floors are $15.60 for one M1, $29.52 for one M4, $74.64 for two M4s plus one M1, or $88.56 for three M4s. A two-hour experiment on a newly allocated host still incurs that minimum; stopping its instance does not release the dedicated host. Explicitly release eligible hosts and clean up remaining resources after experiments. [S169](../sources/S169.md), [S170](../sources/S170.md)

These M1/M4 instances are Mac mini configurations with 16/24 GiB, not the team's M1 Air 8 GB or M4 Pro 16 GB. They change cooling, memory, network and operating environment. AWS reports roughly 6–20 minutes to instance readiness before dependencies, model downloads, data preparation and calibration. Allocate time for service quotas/capacity, compatible AMIs, EBS sizing and private-API/ANE preflight. No current cloud training runtime or ANEForge compatibility was measured. EC2 Mac has no Spot option; long-commitment savings do not fit a short course experiment by default. [S169](../sources/S169.md)

**C. Linux/CUDA workers or simulation — algorithm/control use.** A standard AWS CPU instance can run the reducer and CPU numerical tests; an NVIDIA GPU instance can run a PyTorch/Hugging Face QLoRA port and virtual-client simulations. Neither executes the Apple MLX/ANE path. Separate the port's tokenizer, quantization, dtype and loss semantics before claiming quality equivalence. This research did not obtain a verified current g5/g6 hourly quote or benchmark, so budget it as `workers * quoted_rate * actual_hours + storage + transfer`, and measure time-to-quality. One GPU alternating three logical clients is an algorithm experiment, not three simultaneously training heterogeneous Macs. Spot is an option for eligible Linux capacity if checkpoint/replay semantics are implemented, and is separate from EC2 Mac restrictions.

**Implementation/time judgment:** an SSH-accessible Linux hub is modest infrastructure work; the main engineering effort is the persistent remote-client protocol and fair recipe. Cloud Mac setup adds OS/image/capacity/ANE uncertainty; a CUDA path adds a backend port. Do not book a cloud machine just to discover that the current multi-Mac executable rejects peers. A practical planning allowance is a few days for integration/preflight, followed by measured pilot runtimes to budget the experiment matrix; this is not an observed implementation time.

## 6. Repository gaps to close before the proposal study

These are code-inspection findings, not new benchmark results:

- `execute_federated` in [lora_client.py](../../src/macreduce/lora_client.py) raises for `n_macs > 1`. `run_federated` in [federated.py](../../src/macreduce/federated.py) invokes clients serially in one process; its relay path publishes in process. It is not an independent three-worker training service.
- `MlxLoraTrainer` hardcodes scale 20, defaults to batch 8/accumulation 2, creates fresh AdamW moments per local call and does not carry the saved schedule, clip and recipe through the same path. Its evaluation batch defaults also differ. Unify with the validated curve's recipe before drawing comparisons.
- Each local call rewrites/reloads shard records. Sampler/RNG progress across calls needs an explicit contract; do not assume continued passes merely from increasing a round counter.
- Reducer helpers copy states/deltas to float64. For this adapter, raw fp64 state is 40.37 MB before archive metadata, twice the proposed fp32 payload.
- `encode_tree` reconstructs int8 codes into floating arrays before `pack_state`; the numerical byte counter does not prove the network carries int8 codes. Persist/serialize codes and scales to measure real compression, and separately specify compression of the broadcast direction.
- Star/ring/tree helpers are numerical reductions, not implementations of distinct physical network topologies. Wire-byte formulas are predictions. Measure transport bytes, including retries/metadata; don't report those formulas as captured traffic.
- The local-SGD prototype has a bounded chunk protocol with hash validation, a 64 MiB cap and 256 KiB chunks. Its checkpoint schema is for its model, not automatically a QLoRA adapter/RNG/optimizer protocol. Reuse invariants, not incompatible schemas. Integrate receipts/versioning and outgoing-update durability into the QLoRA worker.
- The federated ANE gate still refuses Qwen3 local training. A successful combined suffix in the curve path does not establish a standalone ANE client, and M4 success does not establish M1 success. Keep GPU, combined, and split cohorts separate; record an unsupported arm as unavailable.

## 7. A bounded experiment plan for three people

**First milestone:** one remote GPU step/round from all three actual machines, correct common initialization, persisted client state, exact byte accounting and fault tests. Start from the CPU numerical oracle; don't treat it as hardware evidence.

**Pilot:** inspect each machine's single-client GPU curve; calibrate device microbatch/accumulation and shard assignments to balance effective-update time; benchmark concurrent transfers; compare two-M4 and three-Mac star federation at K=8,32,128 with one seed and fixed corpus exposure. Include the uniform batch-1 design as a control, not the throughput policy. Run a short true per-step gradient control to verify semantics and characterize the expensive synchronization endpoint. Preserve an end-to-end quality control rather than inferring it from the short timing test.

**Main comparison:** select a small number of K/cohort candidates using validation and network pilots, then repeat those plus the matched single-Mac control over three seeds. Show quality versus tokens and wall time, best-validation checkpoint versus final checkpoint, actual bytes and sync count, straggler/transfer/evaluation breakdown, and failure/replay rate. Predeclare a quality tolerance/target from the control; any specific perplexity threshold is a design choice, not guaranteed reachable on different optimization policies. If the M1 only hurts throughput, report that result.

**Proposal extensions:** only after GPU federation works, compare combined GPU+ANE on each supported device at the same selected K. A standalone ANE client is a separate implementation gate. Then consider true int8 communication, one alternate physical topology, or unequal-K scheduling, one at a time. Do not require all of them for the first three-Mac result.

Suggested ownership, for the human team: one person owns client recipe/data/optimizer parity; one owns networking/coordinator/recovery; one owns M1 calibration/evaluation and experiment manifests. This is a project plan, not a delegation to autonomous agents.

## Limitations, retrieval failures and supersession

M1 memory fit and throughput, all three network paths, thermal behavior, power-sensor coverage, cloud quotas/capacity, cloud ANE access, and statistical convergence remain unmeasured. No federated privacy guarantee follows from merely retaining data locally; the trusted coordinator sees individual deltas, and this plan does not implement secure aggregation or differential privacy.

Recovery record: obsolete Tailscale connection URL failed, recovered through the reference URL and Markdown; guessed arXiv HTML v3 URLs for FLoRA and FedNova failed, recovered with FLoRA v1 HTML and FedNova's publisher PDF; a Flower single-machine URL failed, recovered with its current deployment and TLS guides. Legacy AWS static price URLs returned 403 and several guessed new paths returned 404. Inspecting AWS's published pricing-page script revealed the dedicated-host feed; it was fetched successfully with urllib after handling gzip. The web reader could not open that feed. No verified Linux/CUDA hourly price was recovered; no such rate is asserted. These failures did not yield negative capability claims.

This planning synthesis supplements the earlier [Wi-Fi](../guides/wifi.md), [heterogeneity](../guides/heterogeneity.md), [transport](../guides/transport.md) and [adapter](../guides/optimizers-adapters.md) guides. S162/S163/S175/S176 add newer or deeper readings to S20/S06/S19/S02 respectively; original source snapshots and historical measurements are preserved. It does not upgrade implementation or measured status. See [current state](../project/state.md) and [research gaps](gaps.md).
