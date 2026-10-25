# Feasibility of an ANE-aware distributed training framework for Macs

Research date: 22 October 2026. Prepared as a research brief and design recommendation, not as a course submission. Scope incorporates your clarification: a trainer is the main objective; inference means running one large model across several Macs. The course handout is a guide to project scale, not an assumed binding specification.

**Verdict: a Mac cluster trainer is feasible, and using the Apple Neural Engine for selected training workloads is technically possible. A general, fast, reliable trainer for arbitrary large models across mixed-generation Neural Engines is not an established capability.** The strongest project is a small distributed trainer with infrequent synchronization, a reproducible GPU baseline, and an experimental ANE backend. Large-model sharded inference is feasible separately and already has substantial implementations.

The unresolved question is not whether Macs can exchange tensors. It is whether an ANE execution path improves time or energy to a given model quality after compilation, numerical behavior, CPU work, synchronization, and failures are included.

**What I recommend building:** a coordinator and persistent workers that train a common small language model locally for K steps, combine versioned updates, and repeat. Compare K, backend, and network conditions. Begin with a supported small transformer; make 30B–100B+ distributed inference a separate demonstration using an existing runtime if time permits. Do not make arbitrary large-model ANE training a requirement for success.

## 1. Hardware: the Neural Engine starts at M1

All M-series Mac generations have a Neural Engine, starting with M1 in 2020. M1 and M2 are eligible hardware; “M3 or newer” is unnecessarily restrictive. Apple explicitly lists a 16-core Neural Engine in the original M1 Mac mini. Hardware presence does not imply that a particular model or private runtime works on every generation. [Apple M1 specifications](https://support.apple.com/en-us/111894)

Three capabilities must be recorded separately: ANE hardware, runtime/OS compatibility, and interconnect support.

| Example Mac | ANE | Relevant connectivity | Consequence |
|---|---|---|---|
| M1 Mac mini | Yes | Thunderbolt/USB4, Ethernet | Suitable for a TCP-based worker; not the TB5 RDMA path |
| M4 MacBook Air | Yes | Thunderbolt 4 | Suitable for local ANE/GPU experiments; not TB5 RDMA |
| M4 Mac mini | Yes | Thunderbolt 4; configurable 10Gb Ethernet | ANE availability does not imply RDMA availability |
| M4 Pro Mac mini | Yes | Three Thunderbolt 5 ports | Can participate in the supported TB5 RDMA stack with the required OS/setup |
| M3 Ultra Mac Studio | Yes | Thunderbolt 5 | Demonstrated in Apple's distributed MLX examples |

Sources: [M1 specifications](https://support.apple.com/en-us/111894), [M4 Air specifications](https://support.apple.com/en-ie/122209), [M4/M4 Pro mini specifications](https://support.apple.com/en-us/121555), [Apple distributed MLX session](https://developer.apple.com/videos/play/wwdc2026/233/).

RDMA over Thunderbolt requires Thunderbolt 5 and macOS 26.2 or later, plus explicit enablement. An ordinary USB-C cable, a TB4 port, or being on the same Wi-Fi network does not establish this capability. [Apple JACCL requirements](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/README.md)

The computer accessible in this session reports an **M4 MacBook Air, 24 GB memory, macOS 26.5**, with 40 Gb/s Thunderbolt/USB4 ports and no attached Thunderbolt peer. This was a read-only hardware inspection. It does not establish what other machines you can obtain. This computer can support useful single-node work, but cannot validate a multi-Mac TB5 RDMA experiment.

On newer chips, GPU Neural Accelerators and the separate Apple Neural Engine are different hardware. A faster MLX/Metal workload can use GPU acceleration while leaving the ANE unused. [Apple M5 architecture announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)

## 2. The software situation has changed substantially

| Stack | What it provides | Role in this project |
|---|---|---|
| MLX / MLX-LM | CPU/GPU array computation, differentiation, LLM training and inference | Primary correctness and performance baseline |
| Core ML | Supported model deployment; ANE is one possible execution device | Existing inference/export path |
| Core AI | New model authoring/export/runtime stack for CPU, GPU, and ANE | Evaluate for supported inference on macOS 27+ |
| Direct ANE research runtimes | Private interfaces, graph lowering, selected training implementations | Experimental training backend |
| MLX distributed / JACCL | Communication collectives and launch support | Reuse for the optimized transport baseline |

MLX's advertised execution devices are CPU and GPU. Running an MLX-format Qwen model does not, by itself, use the ANE. MLX-LM already implements distributed inference and fine-tuning. [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm)

**Do not rely on the older statement that Core ML is the only supported inference route.** Apple's Core AI now supplies export recipes, Python authoring components, and Swift runtime utilities. Its documented app-integration requirements are macOS/iOS 27+ and Xcode 27+. Consequently it is not available on the inspected computer's current macOS 26.5 installation. No OS upgrade was performed. [Apple Core AI models repository](https://github.com/apple/coreai-models)

Apple's Core AI guidance distinguishes ANE-oriented static shapes and palettized weights from GPU-oriented linear quantization and more flexible execution. Device preference and successful model conversion are not sufficient evidence of the actual execution placement of every operation; profile the execution. [Apple model deployment guidance](https://github.com/apple/coreai-models/blob/main/skills/skills/working-with-coreai/references/guidance.md)

Core ML does have restricted model-personalization support through MLUpdateTask. Thus “Core ML cannot train anything” is too broad. That API does not establish a general transformer training backend on the ANE. I did not find a supported general-purpose distributed ANE training stack in the Apple material reviewed; Apple positions MLX for distributed and on-device training. [Core ML updatable models](https://apple.github.io/coremltools/docs-guides/source/updatable-model-examples.html), [Apple WWDC26 ML/AI discussion](https://developer.apple.com/videos/play/wwdc2026/8121/)

## 3. Direct evidence that ANE training is possible

The following are primary-source reports and code, **not benchmarks reproduced during this investigation**.

**maderix/ANE:** the training documentation includes Stories110M and Qwen3-0.6B. Its dynamic-weight path passes weights as inputs and compiles reusable kernels once, avoiding the static path's repeated compilation. The implementation still has CPU work, including optimizer/loss-related operations and weight-gradient work. Its documented vocabulary compaction also means its step timings must not be treated as a matched, unmodified Qwen benchmark. This is an important feasibility reference, not evidence of turnkey 30B training. [Training implementation notes](https://github.com/maderix/ANE/blob/main/training/README.md)

**Orion:** the March 2026 paper reports a 110M transformer training for 1,000 steps in 22 minutes on M4 Max and GPT-2 124M inference above 170 tokens/s. It uses private APIs and includes CPU components. Crucially, its discussion reports that its CPU baseline achieved 283 tokens/s on that small inference task and that GPU inference had higher absolute throughput. ANE execution alone does not prove acceleration. [Orion paper](https://arxiv.org/html/2603.06728v1)

**ANEForge:** this later Python runtime provides graph compilation and autograd through private Apple interfaces. Its training documentation distinguishes host-optimizer, device-optimizer, and resident-state paths; it also documents a streamed transformer implementation and remaining numerical/compilation caveats. On one tiny MLP benchmark, its reported ANE time is 0.93 s versus 0.87 s for MLX GPU. That is evidence of working training, not a large-model speed advantage. [ANEForge training documentation](https://github.com/sbryngelson/ANEForge/blob/main/docs/training.md)

The ANEForge paper records a tested chip/OS/compiler combination and explicitly treats the private interface as unstable. Its examples make it a useful research starting point, but support must be checked on each machine. [ANEForge paper](https://arxiv.org/html/2606.17090v1)

A newer architectural study, revised on 15 October 2026, separates measured, reverse-engineered, and predicted claims and reports measurements on M1 and M5. Its abstract also emphasizes that direct access is unsupported and breaks across OS updates. I inspected the abstract and scope, not all 303 pages. [ANE architecture study](https://arxiv.org/abs/2606.22283)

**Interpretation:** ANE training is not physically impossible or merely hypothetical. The remaining research includes model coverage, correctness over long runs, weight exchange, compilation behavior, and useful scaling. A project must say which parts of training execute on ANE, CPU, and GPU; “ANE-assisted training” is often the accurate description.

Avoid making permanent architectural claims from one early implementation. For example, “every weight update requires recompilation” describes some static-weight paths, but is contradicted by later dynamic-input implementations.

## 4. What already exists, and what would be your contribution?

| Existing work | Relevant capability | Implication |
|---|---|---|
| MLX-LM | Distributed inference and LoRA/full fine-tuning | Connecting GPU workers is established |
| EXO | Device discovery, topology-aware model placement, distributed MLX inference | A new Mac inference cluster is not a new category |
| llama.cpp RPC | Remote CPU/accelerator inference; TCP and documented RDMA options | Another useful reference for sharded inference |
| AirTrain | Mac/MLX training using DiLoCo-style infrequent communication | Low-communication Mac training also has prior art |
| ANEMLL | Model conversion and inference targeting ANE through Core ML | ANE LLM inference is already possible |
| ANEMLL Forge | Newer, model-specific Core AI/ANE inference work | Large-model ANE inference is evolving quickly |

[EXO](https://github.com/exo-explore/exo) reports topology-aware placement and tensor-parallel scaling, including large models on four Mac Studios. These are project-reported results; they do not establish the same speedup on laptops or Ethernet.

[llama.cpp RPC](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) exposes remote devices and documents both TCP and RDMA transports. Its maintainers still label the RPC implementation a proof of concept. It is a comparison point, not a general training backend.

[AirTrain](https://github.com/alexandercodes4/AirTrain) explicitly targets MLX training across Macs with infrequent synchronization. Its headline scaling and communication claims were not independently verified here. Its existence is sufficient to rule out claiming that the basic idea of “DiLoCo on Macs” is new.

[ANEMLL](https://github.com/anemll/anemll) offers an ANE-focused conversion and inference pipeline. [ANEMLL Forge](https://github.com/Anemll/anemll-forge) now targets a Qwen3.8-27B adaptation through Core AI, with M5/M6-specific support and substantial compilation, memory, and validation qualifications. Do not confuse this project with sbryngelson/ANEForge, the private-interface Python runtime described earlier.

**A defensible contribution:** measure how backend choice and synchronization frequency interact on heterogeneous Macs. Implement a transparent, reproducible scheduler that chooses or compares ANE/GPU workers and reports time, communication, energy, and held-out quality. A careful negative result—ANE training works but loses after overhead—is a meaningful systems result.

This is an engineering/research contribution, not a claim of a new optimization algorithm or the first distributed Mac framework.

## 5. Where MapReduce fits training

Your intuition fits **coarse training rounds**:

1. Distribute a common checkpoint and disjoint training shards.
2. Each worker performs K local optimizer steps.
3. Each worker returns weights or a delta, token counts, metrics, and the starting checkpoint version.
4. The reducer combines accepted results and publishes the next checkpoint.
5. Repeat, keeping workers and compiled kernels alive.

At K = 1, local parameter averaging can reproduce a synchronous SGD update only under matching initialization, learning rates, weighting, and related assumptions. It is not generally equivalent to averaging gradients before an Adam update. At K > 1, local trajectories diverge; quality becomes part of the experiment.

Traditional file-oriented MapReduce is useful for corpus preparation, evaluation, and independent training branches. Restarting processes and loading or compiling a model for every minibatch would introduce avoidable overhead. Persistent workers retain the simple coordinator/worker design while preserving expensive local state.

**Different distributed methods solve different problems:**

| Method | Communication | Does each Mac need the whole model? | Best fit |
|---|---|---|---|
| Data-parallel gradient averaging | Each synchronization step | Yes, unless combined with sharding | Fast networks, straightforward baseline |
| Local training + periodic averaging | Every K steps | Yes | Ordinary LAN, project-scale trainer |
| Branch-train-merge | After substantial independent specialization | Each branch needs its own model | Domain adaptation and model-merging experiments |
| Pipeline parallelism | Activations and backward gradients between stages | No | Models exceeding one node's memory |
| Tensor parallelism | Frequent layer-level collectives | No | Low-latency interconnects |
| Independent inference replicas | Requests and outputs | Yes | Aggregate serving throughput |

**The critical distinction:** local averaging reduces communication; it does not pool memory for a model that cannot fit on one worker. Training one huge model across small Macs requires model/optimizer sharding and communication through the forward/backward computation, which is a substantially larger project.

DiLoCo supplies an established low-communication training baseline: local AdamW steps and an outer Nesterov update over worker differences. Its paper reports 500-fold less communication in its evaluated setting. That is not a universal Mac speedup or a license to rename arbitrary weight averaging “DiLoCo.” [DiLoCo](https://arxiv.org/abs/2311.08105)

## 6. How the course's model-merging ideas relate

| Reference | What it adds | Suggested use |
|---|---|---|
| Branch-Train-Merge | Independently specialized experts; averaging/ensembling options | Strong alternative trainer workload |
| Branch-Train-MiX | Experts become MoE feedforward components, followed by router training | Stretch goal; adds considerable complexity |
| Task arithmetic | Combine changes relative to a common pretrained checkpoint | Simple merge baseline |
| TIES | Trims small changes and resolves conflicting update signs | One stronger merge comparison |
| Tangent-space analysis | Explains conditions behind successful task arithmetic | Motivation for measuring interference |
| LoRAHub | Composes task adapters using examples from a new task | Relevant if choosing adapter specialization |
| PHATGOOSE | Token/module-level routing among specialized adapters | Alternative inference architecture, not simple averaging |

Primary references: [BTM](https://arxiv.org/abs/2208.03306), [BTX](https://arxiv.org/abs/2403.07816), [task arithmetic](https://arxiv.org/abs/2212.04089), [TIES](https://arxiv.org/abs/2306.01708), [tangent-space paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/d28077e5ff52034cd35b4aa15320caea-Paper-Conference.pdf), [LoRAHub](https://arxiv.org/abs/2307.13269), [PHATGOOSE](https://arxiv.org/abs/2402.05859).

For a small project, choose periodic local training **or** domain-specialized branch-and-merge as the main algorithmic study. Do not implement all seven methods. BTM's own results distinguish domain specialization from random partitioning, so include both IID and domain-separated data only if you can evaluate them properly.

Merges should begin from the same checkpoint, architecture, tokenizer, and parameter ordering. Evaluate the merged model on every domain plus a general held-out set. A reduced training loss on each branch does not establish that their merged weights work well.

One implementation trap: averaging LoRA factors is not the same as averaging their weight updates. In general,

`mean(B_i) × mean(A_i) != mean(B_i × A_i)`.

The left-hand side introduces cross terms. State whether you merge dense deltas, concatenate low-rank updates and optionally recompress, or deliberately use a factor-composition method. Treat optimizer-state retention/reset after a merge as an explicit experimental choice.

## 7. Large-model inference across Macs

This part of the original idea is feasible with existing software. MLX supports distributed operations and MLX-LM includes model-parallel inference; EXO wraps much of the cluster experience. [MLX distributed guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)

For a single model larger than any individual Mac's available memory, **pipeline partitioning** places consecutive layers on different machines. Keep the weights and each layer's KV cache at its owner, send boundary activations, and return the sampled token or equivalent control information. Moving all weights or the complete cache between Macs for every token would defeat the design.

For one autoregressive sequence, the next token generally waits for the current token to pass through the stages. Approximate latency is the sum of stage times plus communication. Pipeline parallelism therefore increases capacity more reliably than it reduces single-request latency. Multiple independent sequences or microbatches can improve stage utilization.

**Tensor parallelism** splits work within layers and can reduce latency, but adds frequent collectives. It makes more sense on compatible Thunderbolt 5 RDMA machines. Inference throughput, first-token latency, and per-token latency must be measured separately.

“Mac-optimized Qwen” can mean MLX-converted/quantized weights, or a different conversion targeting ANE. The official [Qwen3-1.7B-MLX-4bit model](https://huggingface.co/Qwen/Qwen3-1.7B-MLX-4bit) illustrates the former. A model format is not a hardware-placement guarantee. The same distinction applies to larger Qwen variants.

A model's active parameter count is not necessarily its storage requirement. In a mixture-of-experts model, inactive experts still require storage unless explicitly streamed or offloaded. Streaming trades memory capacity for transfer latency.

Distributed ANE pipeline inference is plausible as an integration project, using separately compiled stage models and a transport. It needs correct partition boundaries, compatible tensor layouts, stateful cache management, and measured transfer costs. I did not verify a mature, general multi-Mac ANE-only runtime comparable to the existing GPU-based stack.

## 8. Quantitative limits before implementation

These are **analytical calculations**, checked locally. They are not measured hardware benchmarks. GB means decimal gigabytes; GiB is used explicitly where intended.

**Memory is the first feasibility test.**

| Parameters | Raw 4-bit weights | Raw FP16/BF16 weights | Illustrative mixed-precision Adam state at 16 bytes/parameter |
|---|---:|---:|---:|
| 1B | 0.5 GB | 2 GB | 16 GB |
| 8B | 4 GB | 16 GB | 128 GB |
| 32B | 16 GB | 64 GB | 512 GB |
| 70B | 35 GB | 140 GB | 1,120 GB |

The first two columns exclude quantization metadata, unquantized tensors, cache, activations, and runtime overhead. The training example assumes two-byte weights, two-byte gradients, four-byte master weights, and two four-byte Adam moments. It excludes activations and temporary buffers. Actual memory depends on optimizer, precision, checkpointing, quantization, and sharding; 16 bytes is an illustration, not an MLX-specific guarantee.

LoRA/QLoRA reduces trainable state substantially, but still needs the frozen base and training activations. Inference memory is not a full-training budget. MLX-LM supports low-rank, quantized low-rank, and full fine-tuning. [Fine-tuning documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

For a conventional grouped-query attention cache:

`KV bytes = 2 × layers × sequence_length × KV_heads × head_dimension × bytes_per_element × batch_size`.

An illustrative 28-layer model with 8 KV heads, dimension 128, 8,192-token context, FP16 cache, and batch one needs **0.875 GiB** for KV alone. This is a hypothetical configuration, not a claim about every Qwen model; hybrid architectures require different accounting. Core ML supports stateful models, so persistent state is available, but its placement and runtime costs still need validation. [Core ML stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)

**Gradient communication can dwarf computation.** For a ring all-reduce with n workers and a gradient buffer of G bytes, each worker sends approximately `2(n−1)/n × G` bytes. Four workers synchronizing 1B FP16 gradients therefore send about **3 GB per worker per synchronization**.

| Four-worker case | Bytes sent per worker | Ideal time at 1 Gb/s | At 10 Gb/s | At 20 Gb/s |
|---|---:|---:|---:|---:|
| 110M full FP16 gradients | 330 MB | 2.64 s | 0.264 s | 0.132 s |
| 1B full FP16 gradients | 3 GB | 24 s | 2.4 s | 1.2 s |
| 10M trainable FP32 adapter gradients | 60 MB | 0.48 s | 0.048 s | 0.024 s |

These ideal serialization times assume the stated effective per-link rate and exclude startup latency, reduction, protocol costs, contention, and synchronization. They describe a ring, not a central gather/broadcast reducer. A coordinator can have a more severe aggregate bottleneck. A Wi-Fi PHY rate is not a sustained per-worker transfer rate.

For local averaging, the analogous round cost uses the transmitted weight/delta dtype and includes download as well as upload; it cannot blindly reuse the gradient-buffer size. If one such round takes C seconds every K steps, its amortized cost is C/K per step. Larger K improves that term but can harm convergence.

A 2026 study of four Mac Studios reports roughly 20 Gb/s per TB5 link for its measured RDMA path, well below nominal interface bandwidth. It reports about 10 microseconds for small-message operations and improvements from multiple links for large training transfers. Its headline fine-tuning improvement is against a particular single-trunk, non-overlapped baseline, not a universal cluster speedup. [RDMA training study, revised October 2026](https://arxiv.org/html/2609.18066v3)

**Do not equate a cable label with useful training bandwidth.** Measure small-message latency and large-message bandwidth independently.

## 9. Interconnect and topology decisions

For the project, ordinary Ethernet is a useful starting point. Wi-Fi is worth testing as a constrained condition if the machines are already available. Use infrequent synchronization rather than expecting either to behave like an accelerator interconnect.

For an optimized baseline on compatible hardware, use MLX's TCP ring or JACCL. There is a documentation discrepancy worth preserving: the high-level MLX guide says JACCL requires a fully connected mesh, while the newer JACCL source README also documents ring topology. Pin a version and verify the topology that version actually supports. Do not design around unversioned assumptions. [MLX guide](https://ml-explore.github.io/mlx/build/html/usage/distributed.html), [JACCL source documentation](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/README.md)

In a full mesh, n machines need n(n−1)/2 direct links and n−1 relevant ports per machine. Four machines therefore need six cables. A simple physical chain is not a full mesh. A ring is easier to cable but has different collective behavior. A Thunderbolt dock should not be assumed to act as a transparent multi-host RDMA switch.

Unified memory is local to each Mac. Multiple connected Macs do not become one cache-coherent memory system; the runtime still owns placement, communication, and synchronization. Similarly, ANE and GPU work can compete for local memory bandwidth and the thermal/power envelope. Additional utilization is valuable only if useful throughput or energy efficiency improves.

## 10. Proposed framework architecture

Keep a shared control layer with separate execution modes for training replicas and sharded inference.

| Component | Minimal responsibility |
|---|---|
| Coordinator | Worker registration, model/version manifest, data assignment, round membership, checkpoints, metrics |
| Persistent worker | Load model once; train K steps; export/import state; report actual backend and timing |
| GPU backend | MLX reference implementation with the same model, data, loss, and optimizer semantics |
| ANE backend | One pinned research runtime and one validated model configuration |
| Transport | SSH/SCP baseline for coarse jobs; optionally persistent TCP or existing collectives |
| Reducer | Weighted averaging or one explicitly implemented outer optimizer |
| Evaluator | Held-out loss/perplexity by domain, numerical comparisons, timing and energy accounting |

Suggested backend contract: `load`, `train_local`, `export_state`, `import_state`, `evaluate`, and `capabilities`. Export a canonical parameter order and dtype so the reducer does not depend on backend-specific packing. The ANE backend must validate that importing new weights actually changes subsequent execution, rather than leaving stale compiled constants.

Every result should include run ID, round ID, base-checkpoint hash, worker ID, completed steps/tokens, parameter schema, and checksum. Write results atomically. Commit each worker's accepted result at most once. For the first version, freeze round membership; retry or restart a failed round instead of quietly merging stale checkpoints. Preserve data-loader and random-generator state for resumption.

Start heterogeneous workers with the same algorithm and batch/token accounting. Weight equal-objective updates by actual token/sample contribution when appropriate. Changing local work to favor faster devices changes the optimization behavior and deserves its own controlled experiment.

If using ANE for a frozen base and GPU for LoRA, remember that adapters inside a transformer still require gradients through subsequent frozen operations. A forward-only ANE model is insufficient for arbitrary internal-adapter training unless the needed backward computations are supplied. A frozen feature extractor with a trainable final head is a simpler, narrower case.

## 11. A project that fits the supplied time frame

There are 15 days from this research date to the stated November 6 proposal deadline and 43 days to December 4. The following is a proposed scope, not scheduled work.

**Research question:** When does infrequent synchronization make distributed language-model training useful across heterogeneous Macs, and does an ANE backend improve energy or wall-clock time to a fixed validation quality compared with GPU training?

**Core deliverable:** a two- or three-Mac trainer for a small causal transformer, with checkpoint/retry support, a GPU baseline, and an ANE experiment. Target roughly 20M–110M parameters only after the selected backend passes correctness and memory checks; start with a much smaller block/model to validate the integration. Qwen3-0.6B is a possible extension, not the starting requirement.

**Choose one meaningful extension:** either domain specialization plus a merge baseline, or adaptive synchronization/worker assignment. A larger Qwen LoRA experiment can demonstrate practical relevance if resources permit. Large-model inference should remain independent of the training success criterion.

| Dates | Evidence to obtain |
|---|---|
| Oct 22–28 | Inventory two/three actual Macs; reproduce a local GPU run; verify the selected ANE training path and state import/export |
| Oct 29–Nov 5 | Two-worker training rounds; checkpoint/restart; one honest scaling result and one numerical comparison |
| Nov 6 | Proposal with measured feasibility and a clear fallback |
| Nov 7–18 | Fixed-K communication experiments and equal-token/quality comparisons |
| Nov 19–26 | Heterogeneous/network experiments; selected extension; energy measurements if instrumentation is available |
| Nov 27–Dec 3 | Repeated runs, figures, limitations, reproducibility checks, writing in the team's own voice |
| Dec 4 | Final deliverable under the handout's deadline |

If ANE correctness or useful scale is not demonstrated before the proposal, choose explicitly between a smaller ANE characterization project and a GPU-based distributed-training project with a documented ANE feasibility result. Do not silently claim a GPU project exploits the ANE.

The supplied Track 1 specifically names AWS. A Mac adaptation or hybrid of the two tracks has not been established as course-approved by this research; check that narrow question with the instructor before selecting the official track.

## 12. Measurements and decision gates

**Gate A — local correctness.** Match forward loss and gradients against a reference at fixed weights and inputs. Use error tolerances justified by precision, then run at least 1,000 steps or a predeclared token budget. Test save/resume and imported-weight replacement. A single successful matrix multiplication is insufficient.

**Gate B — distributed correctness.** Compare one-worker execution with two workers on controlled data. Verify reduction weighting, checkpoint version checks, and retry idempotency. For a simple SGD configuration, use K = 1 to establish expected equivalence. For Adam or local training, compare trajectories statistically rather than requiring false mathematical equivalence.

**Gate C — useful scaling.** Compare the best single Mac, all-GPU cluster, all-ANE-assisted cluster, and a mixed backend configuration if feasible. Test K = 1, 10, 50, and 100 initially. Record total tokens, not just steps, and do not let additional hardware appear better solely because it consumed more training data.

**Gate D — quality.** Report validation loss versus tokens and versus wall time, with fixed held-out sets. For specialized branches, report per-domain and general performance before and after merging. Never tune merge weights on the final test split. Use several seeds where affordable, and show uncertainty.

**Gate E — end-to-end costs.** Separate initial conversion/compilation, warm training, state export/import, serialization, transfer, reduction, evaluation, and restart. Report memory peaks, swap, and sustained behavior after warmup. Measure whole-system energy if possible; ANE-only power excludes CPU, memory, network, and other costs.

Proposed practical thresholds, chosen as project decisions rather than published standards:

- Keep an ANE optimization only if it preserves the selected quality tolerance and offers a meaningful end-to-end benefit, such as at least 20% less energy at comparable quality or improved throughput under a defined concurrent workload.
- Call distributed training useful if it improves time to the same validation target over the fastest available single machine. A target of at least 1.3× on two machines is a reasonable initial hurdle, not a predicted result.
- Treat a capacity-only inference result honestly: “runs a model that does not fit on one node” can be valuable even when latency increases.
- If a backend works only for a fixed chip/OS/model combination, make that the supported experimental scope.

A GPU+ANE experiment should compare GPU alone, ANE alone, and concurrent execution under identical work accounting. Do not infer additive throughput from isolated benchmarks.

## 13. Limitations and what remains unverified

This investigation retrieved official Apple documentation/transcripts, project source documentation, and research papers. Repository claims and paper results were distinguished from API guarantees. Some sources are moving main branches; freeze commits and model revisions before experimentation.

No model was downloaded, no training job was run, no private ANE runtime was installed, and no multi-node benchmark was performed. I deliberately did not build a framework prototype: existing evidence establishes basic possibility, while a one-Mac prototype would not answer the central scaling question. The local work consisted of read-only hardware inspection and checking the arithmetic above.

The precise inventory of the other Macs, effective network bandwidth/latency, dataset, model, acceptable validation degradation, and available measurement equipment remain unspecified. Those determine actual throughput and feasible size. Nothing in this brief establishes a cost advantage over renting a GPU or buying another machine.

Two retrieval limitations were handled without treating missing content as evidence: Apple's RDMA technical-note page did not render fully through the web reader, and a direct download was blocked by network resolution in the execution environment; the official WWDC transcript and JACCL documentation supplied the relevant requirements instead. The long ANE architecture paper exceeded the reader's HTML limit; only its abstract/scope is used here.

**Decision:** proceed with a narrowly scoped distributed trainer and an explicit ANE research experiment. Reuse existing execution and communication libraries where the project rules allow, while implementing your own transparent coordination/reduction layer. Treat fast, arbitrary, large-model ANE training as the longer-term research ambition. Treat large-model GPU-based inference across Macs as an already feasible, separate capability.
