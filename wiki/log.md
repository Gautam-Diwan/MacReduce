---
title: "MacReduce wiki change log"
type: log
status: active
updated: 2026-10-27
sources: ["S01"]
---

# MacReduce wiki change log

This log is append-only. Corrections get new entries rather than rewriting earlier decisions or results.

## 2026-10-22 — Project relocation and initial wiki

Moved the original prototype to the course project's `MacReduce` directory. Renamed the package, command, environment variables, and active documentation. Preserved original experiment bytes and archived proposal artifacts; the migration manifest records original hashes. The active proposal is Markdown under `docs/proposal.md`; archived Word documents retain historical branding.

Adopted Karpathy's local LLM-wiki pattern with immutable source snapshots, editable synthesis, an index, maintenance schema, and this log. Retrieved 22 sources; their UTC timestamps fall on October 23 because retrieval occurred after midnight UTC. Added implementation guides, evidence boundaries, staged experiments, decisions, source cards, and conditional novelty candidates.

The gist page/API, AirTrain main-branch path, and initial MLX-LM LoRA path needed fallback retrieval. See [the ledger](research/gaps.md). Paper reading scope is mostly abstract-only; this is an initial research map, not an exhaustive review. Structural lint and renamed-code validation are recorded in subsequent entries when completed.

## 2026-10-23 — Rename and wiki validation

The renamed package imports from the canonical course folder. The CPU test suite passes 25 tests with one intentionally skipped opt-in MLX test. A new two-worker loopback run completed eight rounds and reproduced the original validation loss (6.058714 to 1.768063). All 31 historical evidence-file checksums match the migration record. Raw source hashes and all 48 wiki pages passed structural lint. No physical Wi-Fi, multi-Mac, ANE, or M5 performance claim was added.

Added intake, search, and structural lint tools with tests for immutable source copying, duplicate rejection, provenance, path containment, and link/index integrity. Source URL mutability flags were conservatively corrected to true because these URLs do not guarantee immutable revisions; snapshot bytes and hashes were unchanged. Refined the active proposal with conditional research candidates and links to implementation gates. Paper reading remains mostly abstract-level; nearest-method full-paper review is unfinished.

## 2026-10-24 — MLX measurement, model config, and chunked artifacts

Measured the MLX GPU path on this M4: doctor smoke returned 4096.0, the opt-in parity test passed for the default model and a 2-layer config, and a 3-round loopback demo finished. `xctrace` was unavailable, so there is no GPU capture. The extended gradient audit and 100-step check were not run. Raw outputs are in [evidence/mlx-validation](../evidence/mlx-validation/README.md).

Added an immutable `ModelConfig` and repeated Transformer blocks. The default configuration still matches the original 29,888-parameter checkpoint bytes. Checkpoints and run state carry the config. Replaced the 2 MB response ceiling with a 64 MiB chunked artifact protocol: manifest, chunk hashes, size check before allocation, temp file, and atomic complete only after hashes match. Duplicate retries do not apply an update twice. Partial and corrupt uploads are rejected. Compression, a real corpus, a 1M-parameter model, and physical Wi-Fi remain undone.

The default suite passed 33 tests with 2 opt-in MLX tests skipped. Those two tests passed when enabled. `ruff check .` passed. This entry records that structural review; it does not claim the unfinished MLX or scaling gates.

## 2026-10-25T04:12:57.015352+00:00 — Intake S23

Added [S23](sources/S23.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.042026+00:00 — Intake S24

Added [S24](sources/S24.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.064272+00:00 — Intake S25

Added [S25](sources/S25.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.085979+00:00 — Intake S26

Added [S26](sources/S26.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.107538+00:00 — Intake S27

Added [S27](sources/S27.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.129013+00:00 — Intake S28

Added [S28](sources/S28.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.152461+00:00 — Intake S29

Added [S29](sources/S29.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.174762+00:00 — Intake S30

Added [S30](sources/S30.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.197710+00:00 — Intake S31

Added [S31](sources/S31.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.220448+00:00 — Intake S32

Added [S32](sources/S32.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.243483+00:00 — Intake S33

Added [S33](sources/S33.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.267871+00:00 — Intake S34

Added [S34](sources/S34.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.290438+00:00 — Intake S35

Added [S35](sources/S35.md) as unread; source synthesis is pending.

## 2026-10-25T04:12:57.312833+00:00 — Intake S36

Added [S36](sources/S36.md) as unread; source synthesis is pending.

## 2026-10-25 — Scope correction, source refresh, and decision 0003

The user rejected the Wi-Fi local-SGD framing. Fetched and read primary READMEs and docs for the systems named as prior art and intook them as S23–S36 (one intake each, entries above): maderix/ANE top README (S23, contains the GPU-prefill to ANE-decode IOSurface claim that S11 does not), Hypura (S24), AtomGradient hybrid ANE/MLX bench (S25), rs-ane-infer (S26), ANE Fieldfare (S27), RavenX distributed LoRA over JACCL (S28), laya-apple (S29), BTankut DGX Spark plus Mac Studio P/D over ZeroMQ (S30, resolved by repository search), Mac Spark playbook (S31), `mlx_lm/generate.py` speculative path (S32, read not run), an mlx-lm issue search with four issue bodies read (S33), MLX tensor- and data-parallel example pages (S34, S35), and the coremltools compute-units section (S36). Each source card states inspected scope, findings, limitations, and the claim it blocks. All numbers are source-reported; none were reproduced.

Re-read S02 and S18 without changing bytes. The uploaded `distributed-0.html` is a text rendering of the same MLX 0.32.4 page as S02 and was not intaken. The EXO README on `main` was byte-identical to S18; API metadata shows the repository active and not archived. Retrieval failures recorded in the gaps ledger: two guessed MLX example URLs (404, corrected from S02 link targets), `pipeline_generate.py` (404), and Apple Developer pages that return a JavaScript shell.

Wrote [decision 0003](decisions/0003-build-target.md): build a per-request phase-placement scheduler for LLM serving (drafter engine, speculation depth, prefix-KV source) plus an A-B-B-A benchmark against the strongest static configuration, on the available M4, with a second Mac only for the remote prefix arm. Marked decision 0001 superseded. Rewrote [novelty](research/novelty.md) with the Wi-Fi study demoted and the reasons; rewrote [related work](research/related-work.md), [gaps](research/gaps.md), [roadmap](project/roadmap.md), and [state](project/state.md); added scope notes to the Wi-Fi, inference, ANE, experiments, and architecture pages; updated the index. No scheduler code was written and no new measurement was taken; the trainer remains the measured prototype. Structural lint passed with 63 pages and 36 sources. Literature searches on adaptive speculation depth and KV reuse policies are still owed before any novelty wording.

## 2026-10-25 — Placement scheduler code

Added a tested placement rule and an A-B-B-A harness. The rule chooses drafter engine, depth in {0, 1, 2, 4, 8}, and prefix source from measured costs. An unverified ANE cost cannot win. A missing link rate is not treated as a free transfer. `macreduce equivalence` then ran on this M4 with `mlx-lm==0.32.2` and `mlx==0.32.4`. Greedy speculative decoding at depth 2 matched greedy decoding for 16 tokens on the short, long, and code prompts. All three hit the token cap. That is not a timing result or a depth sweep. Record: [evidence/serve-equivalence](../evidence/serve-equivalence/README.md). CPU and ANE drafters have no executor. The local-SGD trainer is unchanged.

## 2026-10-25 — One-M4 placement study

Ran `macreduce bench` with three fresh-process cycles, 32 tokens, and the pinned Qwen2.5 1.5B/0.5B 4-bit pair. All 45 depth trials matched greedy tokens and all hit the length cap. Short prompts were fastest with no draft. Long prompts decoded fastest at depth 4 (about 137 tok/s versus about 96 with no draft) and reached the first token fastest with no draft. Code decoded fastest at depth 2; the cost model chose depth 4 and was slower on the held-out passes. A CPU-loaded draft matched the short prompt and was slower than no draft. Reloading a just-written prefix cache matched recomputed tokens; that load time is a warm page cache, not cold storage or a network. `coremltools` is absent, so the ANE gate did not run. No second Mac was available. Per-step depth changes were not implemented. Record: [evidence/serve-benchmark](../evidence/serve-benchmark/README.md).

## 2026-10-25 — Qwen3 pin and Core ML plan probe

Replaced the serving pin. Qwen2.5 evidence was left in place. Gemma 4 E4B and E2B 4-bit are about 8.8 GB together; `df` showed 3.0 GiB free on this volume, so they were not downloaded. Qwen3.5 was not pinned: the installed `mlx-lm` 0.32.2 `qwen3_5.py` `make_cache` uses `ArraysCache` for linear-attention layers, and that cache reports `is_trimmable()` false, which stock speculative decoding rejects. Qwen3 has no `make_cache`, so generation builds a `KVCache`, which is trimmable.

Pinned `mlx-community/Qwen3-1.7B-4bit` at `3b1b1768f8f8cf8351c712464f906e86c2b8269e` and `mlx-community/Qwen3-0.6B-4bit` at `73e3e38d981303bc594367cd910ea6eb48349da8`, both last modified 2025-04-28. The loader now passes those revisions. On this M4, greedy speculative decoding at depths 1, 2, and 4 matched greedy decoding for 32 tokens on short, long, and code. Depth 8 matched long and code and mismatched the short prompt at token 26. No new speed sweep was run. Record: [evidence/serve-equivalence-qwen3](../evidence/serve-equivalence-qwen3/README.md).

Installed `coremltools==9.0` into the serve extra. `macreduce ane` converts a float16 3-to-64 convolution and reads `MLComputePlan` for `CPU_AND_NE` and `CPU_ONLY`. The plan assigns the convolution and ReLU to `MLNeuralEngineComputeDevice` under the first constraint and to `MLCPUComputeDevice` under the second. Twenty predictions took about 0.038 s either way, so wall time does not confirm residency. `powermetrics` refused without superuser, and ANE watts stay null. The draft model was not converted, so `placement_verified` stays false. coremltools says Torch 2.7.0 is the newest version it has tested; this environment has Torch 2.14.1 and the conversion still produced an mlprogram. Record: [evidence/ane-probe](../evidence/ane-probe/README.md).

## 2026-10-25 — Gemma 4 pair and mactop during the convolution

The Hugging Face cache directories for `mlx-community/gemma-4-e4b-it-4bit` and `mlx-community/gemma-4-E4B-it-assistant-bf16` contained the README only. The weights were downloaded at `475b9088d29754a3379866cf5aeb6b41acd313c2` and `844e008e06ef5562bdb89428d851d0634d119dcd`. `mlx-lm` 0.32.2 rejects the assistant with `Model type gemma4_assistant not supported.` The model card describes that file as a 4-layer MTP drafter for `mlx_vlm`, paired with the 15.91 GB bf16 E4B, which was not downloaded because 7.1 GiB remained free. `mlx_vlm` was not installed. The 4-bit target alone, on one raw 32-token prompt, decoded at 33.93 tok/s on the second pass. The same call on Qwen3 1.7B 4-bit decoded at 94.46 tok/s. The serving pin stays Qwen3. Record: [evidence/gemma4-local](../evidence/gemma4-local/README.md).

`macreduce ane` now also runs `mactop` v2.1.5, because `powermetrics` still requires superuser. On a repeat after the download finished, the first `CPU_AND_NE` sample reached 0.607 W and 4.81 GB/s of ANE bandwidth. The later samples in that 2-second window matched the idle floor. `CPU_ONLY` stayed at 0 W and 0 GB/s. `ane_active` stayed 0. The reading is system-wide. `placement_verified` stays false. Record: [evidence/ane-probe](../evidence/ane-probe/README.md).

## 2026-10-25 — Qwen3.5 2B speculative check, Gemma cache removed

Removed the Gemma 4 revisions downloaded for the previous check. The Hugging Face cache scan freed 5.371 GB: `gemma-4-e4b-it-4bit` at `475b9088d297` and `deb1db712068`, and `gemma-4-E4B-it-assistant-bf16` at `844e008e06ef`.

Loaded `mlx-community/Qwen3.5-2B-4bit` at `674aaa7240b91e8012fcad5d791b7dfe5ba90207` with draft `mlx-community/Qwen3.5-0.8B-4bit` at `da28692b5f139cb0ec58a356b437486b7dac7462`. The 2B cache is 18 `ArraysCache` layers and 6 `KVCache` layers. Plain generation ran. Speculative decoding raises `Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'}).` The 4B repo `mlx-community/Qwen3.5-4B-4bit` at `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` was checked from its config: same `qwen3_5` class, 32 layers, `full_attention_interval` 4, so 24 linear layers. Its weights were not downloaded. The serving pin stays Qwen3. Record: [evidence/qwen35-local](../evidence/qwen35-local/README.md).

## 2026-10-25 — Qwen3 sweep and a multi-sample ANE record

`macreduce model-check` now loads the serving pair, records cache kinds when the model exposes `make_cache`, and writes one plain run plus one draft run. The short-prompt check on the Qwen3 pin matched at depth 2.

`macreduce bench --cycles 3 --max-tokens 32` then ran that pin. All 45 trials hit the length cap. Depths 0 through 4 matched greedy tokens. Depth 8 mismatched the short prompt at token 26 on every replicate, so those rates are absent. Exact depths decoded fastest at depth 2 on short (95 tok/s), long (99 tok/s), and code (95 tok/s). Time-to-first-token was fastest at depth 0 in every segment. The cost model did not beat hindsight on both metrics. A CPU draft on the short prompt matched and ran once. The warm prefix reload matched. Record: [evidence/serve-benchmark-qwen3](../evidence/serve-benchmark-qwen3/README.md).

`macreduce ane` was run again on a larger two-convolution network. The compute plan still assigns the convolutions and ReLUs to `MLNeuralEngineComputeDevice` under `CPU_AND_NE` and to `MLCPUComputeDevice` under `CPU_ONLY`. `mactop` v2.1.5 recorded nine consecutive samples at 0.581–0.802 W and 10.23–11.89 GB/s during the `CPU_AND_NE` loop (424 predictions in 8.00 s). The `CPU_ONLY` loop, 198 predictions in 8.04 s, stayed at 0 W and 0 GB/s for all 40 samples. Later samples in the `CPU_AND_NE` capture returned to the idle floor. `ane_active` stayed 0. `powermetrics` still requires superuser. The serving draft was not converted. Record: [evidence/ane-usage](../evidence/ane-usage/README.md).

## 2026-10-25 — Decoder block ranked across CPU, GPU, and the Neural Engine

`macreduce ane-workload` converts one float16 decoder block (batch 1, 128 tokens, width 256, four heads) and times it on CPU-only Core ML, CPU+Neural Engine, and PyTorch MPS. The compute plan assigns the linear and matmul ops to `MLNeuralEngineComputeDevice` under `CPU_AND_NE` and to `MLCPUComputeDevice` under `CPU_ONLY`. The largest absolute difference from the float32 PyTorch forward was 0.00307. Measured seconds per forward were 0.000168 on CPU, 0.000186 on the Neural Engine path, and 0.000666 on GPU. `choose_device` selected CPU. Three mactop samples during the Neural Engine loop read 1.54–14.22 GB/s; the CPU-only loop stayed at 0. Later samples in the Neural Engine capture returned to the idle floor. A concurrent GPU and Core ML run produced 13 overlapping samples. The Qwen draft weights were not converted, and ANEForge and Orion were not pinned: this is the Core ML inference path. Record: [evidence/ane-decoder](../evidence/ane-decoder/README.md).

## 2026-10-25 — Full-depth forward and the two-sample gate

A rerun of `macreduce ane-workload` measured the decoder at 0.000170 s on CPU+Neural Engine and 0.000176 s on CPU. mactop recorded two samples at 14.15 and 14.30 GB/s, then the capture returned to the idle floor. The gate required three samples, so `placement_verified` stayed false and the ranking omitted the Neural Engine time. The gate now accepts two samples. Record: [evidence/ane-decoder](../evidence/ane-decoder/README.md).

`macreduce infer` generated with the pinned Qwen3 pair on the short prompt. Plain decode was 83.0 tok/s. Speculative depth 2 was 92.7 tok/s and matched those tokens. A 28-layer network at the Qwen3-0.6B widths, 596,042,752 random parameters, compiled once with 1,213 Neural Engine ops and 0.0207 s per forward. The command could not save that package: the data volume is full, and the compiler copies the 1.1 GB weight file. The conversion now stops when less than 3 GB is free. Record: [evidence/infer-qwen3](../evidence/infer-qwen3/README.md).

## 2026-10-26 — Private ANE APIs allowed, WikiText-2 QLoRA baseline

Decision 0003 no longer bars private ANE APIs. An ANE training stack is in scope when it addresses a gap ANEForge, Orion, or maderix leave open. The training baseline is QLoRA of `mlx-community/Qwen3-0.6B-4bit` on Salesforce WikiText-2 raw v1, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The reusable module is `src/macreduce/finetune.py`, command `macreduce lora`. Perplexity is the exponential of the mean token loss, reported on the test split before and after the adapter.

## 2026-10-27 — Course proposal rewritten to one page

The open-ended proposal is [docs/proposal.md](../docs/proposal.md), with a one-page PDF beside it. Dataset: Salesforce WikiText-2 raw v1 at `b08601e04326c79dfdd32d625aee71d232d685c3`. Metric: full-split test perplexity. Baseline: one-epoch QLoRA of the pinned Qwen3-0.6B-4bit on the MLX GPU. The Neural Engine comparison is gated on residency and loss agreement. The October 23 Wi-Fi text is [archived](../docs/archive/proposal-2026-10-23.md). Course dates used here are feedback 2026-10-28, teams 2026-11-02, check-in 2026-11-18, and report 2026-12-01. The one-epoch perplexity and the training-step gate are not yet measured.

## 2026-10-27 — Distributed training and explicit Neural Engine stacks

The 10-805 proposal, lead Gautam Diwan (gdiwan), now asks three things on the same WikiText-2 QLoRA setup. Does an explicit ANEForge, Orion, or maderix training step beat the GPU baseline on perplexity or step time? Should the Neural Engine run inference while the GPU trains? Do a parameter server, local SGD, or a non-reducing inference seat beat ring all-reduce once a second Mac exists? `src/macreduce/train_plan.py` and `macreduce train-plan` write that list. The probe marks Orion and maderix unpinned. A missing import is not replaced with the GPU. No private stack was installed or run. Multi-Mac arms are unmeasured. The one-page proposal was rewritten to this question.

## 2026-10-27 — Heterogeneous all-reduce and a three-epoch curve

Search on this date, abstracts only, not ingested as source cards: BlueConnect decomposes all-reduce across a fast tier and a slow tier. Blink packs trees when link capacity is unequal. BytePS argues a flat all-reduce is no longer optimal once another device can sum gradients. MLX documents a TCP ring that is always available and a JACCL backend that needs Thunderbolt 5. This machine has neither a second Mac nor Thunderbolt 5.

The course plan now reports a single-Mac table and a multi-Mac table. The single-Mac baseline is GPU QLoRA. The single-Mac heterogeneous arm is a GPU rank plus an explicit Neural Engine rank in one reduction. The multi-Mac baseline is MLX ring all-reduce over GPU ranks. The other multi-Mac reductions are a flat heterogeneous all-reduce, a within-Mac sum followed by a ring, and a tree on four ranks. A mixed ring is rejected because a ring assumes equal ranks.

The fine-tuning curve is three epochs on the first 4,096 training lines, batch size 8, gradient accumulation 2. Validation and test stay the full WikiText-2 splits. The model stays the pinned Qwen3-0.6B-4bit. `macreduce lora` uses that schedule. The curve has not been run.

## 2026-10-27 — Heterogeneous federated learning

The course question is heterogeneous federated learning. The project name stays MacReduce. A client is a GPU, a GPU and Neural Engine combined, or those devices as two clients. Every Mac in a run uses the same mode and a shard of one WikiText-2 mixture. The payload is the LoRA adapter delta. FedAvg is the communication reduction. A per-step all-reduce is the control. Star, ring, and tree must match on uncompressed deltas. Unequal step counts use the FedNova normalization from S06, whose card is an abstract plus one equation, not a full-paper review. Int8 quantization keeps an error residual.

`src/macreduce/federated.py` and `macreduce federate` implement the cohort, the shard, the averages, FedNova, and the quantizer. Tests cover those identities. No training run was started. No private stack was installed. FetchSGD and hierarchical quantized averaging were found by search on this date and are not source cards. Serving placement and speculative decoding remain in the tree as future scope. The one-page proposal was rewritten to this rubric. Multi-Mac cells are still unmeasured.

## 2026-10-27 — Federated round loop, still not trained

`run_federated` now runs FedAvg and per-step all-reduce on named adapter tensors, with int8 error feedback, FedNova, and star, ring, and tree means. A second Mac raises `PeerRequired` unless a relay is passed. The loopback relay was tested with synthetic payloads. It is not a second Mac. `src/macreduce/lora_client.py` holds the MLX QLoRA local step and `require_ane`, which refuses a missing or uncompiled Neural Engine stack. `macreduce federate --execute` calls that path. It was not invoked. The CPU reference in the tests is SGD toward a fixed target, not WikiText. The proposal was rewritten in the shape of a one-page NeurIPS preprint, with WikiText-2, validation perplexity, and one-Mac GPU FedAvg named in the evaluation section. The official NeurIPS style file was not fetched. The PDF uses the same text width, Times text, abstract, and numbered sections.

## 2026-10-27 — ANEForge step on the engine

`aneforge==0.4.0` is installed in the project environment. `macreduce ane-forge` does `import aneforge as af`, runs four Adam steps of a 16-by-1 linear map, and keeps a 256-wide matmul busy. Forward, backward, and update use device mask `0x4`, the package's Neural Engine mask. During the matmul, system mactop showed Neural Engine power up to 0.79 W and combined bandwidth up to 2.9 GB/s. `ane_active` stayed 0. mactop is not per-process. Raw result: [evidence/ane-forge](../evidence/ane-forge/README.md). Orion still has no Python import. This is not a Qwen3 LoRA step.

## 2026-10-28T23:15:29.616023+00:00 — Intake S37

Added [S37](sources/S37.md) as unread; source synthesis is pending.

## 2026-10-28 — Read S37

Read the abstract, chapter 1 through the M1 envelope, the opening of chapter 6, chapter 14 sections 14.4 through 14.10, and chapter 15 of [S37](sources/S37.md). The PDF is 303 pages. The rest was not read. The card records the fp16 datapath, the direct `e5rt` route, and training by a host-built backward graph. Silicon figures in those chapters are the author's M1 and M5 measurements. [Related work](research/related-work.md) and the [ANE guide](guides/ane.md) link the card. This is not a claim about Qwen3 on the project M4.

## 2026-10-28 — One-Mac feasibility probe

`macreduce feasibility` ran three traces on this M4: GPU QLoRA, an ANEForge matmul, and both at once. MLX peak memory was about 1.3 GB under an 8 GB limit. GPU-only Neural Engine power was 0. The engine-only trace peaked at 0.76 W and 3.28 GB/s. One combined sample showed 15.9 W on the GPU and 0.73 W on the engine together. The engine graph is not Qwen3, so the arms do not share a perplexity or a delta. The 4-step, 2-batch perplexity is too short to be the proposal curve. Raw files: [evidence/feasibility](../evidence/feasibility/README.md). Report: [one-Mac feasibility](experiments/2026-10-28-one-mac-feasibility.md).

## 2026-10-28 — Curve figures

`macreduce lora` now traces mactop, including the Prometheus scrape, and writes matplotlib figures for loss, validation perplexity, power, cumulative energy, memory, and latency. `src/macreduce/plots.py` and `src/macreduce/telemetry.py` hold that code. `macreduce figures` redraws a saved run. Matplotlib and seaborn are project dependencies, added with uv. The three-epoch Qwen3-0.6B curve is the run those figures are for. The short feasibility probe is not that curve.

## 2026-10-28 — Working set, tqdm, checkpoints

The QLoRA entry is `python main.py --config configs/qlora_m4.yaml`. The memory budget is the GPU's recommended working set (`infrastructure.memory_limit_bytes: auto`), and the wired limit is raised to that size. An earlier search that capped MLX at 8 GB was stopped. `training.batch_size: auto` now keeps the microbatch with the highest steady tokens per second. The uncheckpointed step is tried first. Gradient checkpointing is only the fallback when that step does not fit. Accumulation is 1, so one microbatch is one AdamW update. The base stays Qwen3-0.6B 4-bit with rank-8 LoRA. The Qwen3 backward graph is still not compiled for the Neural Engine, so this step stays on the GPU. The training loop uses tqdm. Each epoch writes a checkpoint with the adapter, AdamW moments, and the step under `checkpoints/`. `--resume latest` continues that checkpoint. The three-epoch curve has not finished. The power figure is instantaneous watts, not cumulative joules.

## 2026-10-28 — Batch search starts small

A batch-16 step with no activation recomputation peaked at 19.08 GB, just over the 19.07 GB working set, at 171 tokens/s. That is the memory ceiling, not a fast step. The search now starts at batch 1 and stops when a larger batch drops below 90% of the best tokens/s. The running search was restarted so it uses that rule.

## 2026-10-28 — Curve started at batch 2

The search that this run used measured, with no activation recomputation: batch 1 at 285 tokens/s and 0.90 GB, batch 2 at 350 tokens/s and 2.92 GB, batch 4 at 220 tokens/s and 4.28 GB. It kept batch 2. An earlier pass the same evening ranked batch 1 above batch 2, so three probe steps are noisy. The live process is the second ranking. Baseline scoring of the full test split had started. The three-epoch result is not in yet. Checkpoints are every epoch.

## 2026-10-28 — Timestamped GPU and Neural Engine runs

The GPU-only curve was stopped during its baseline pass. Each `python main.py --config configs/qlora_m4.yaml` launch now creates a new directory under `evidence/qwen3-0.6b-curve/` named with the local timestamp, and figures are written only inside that directory. `placement.mode` is `gpu_ane`. The Qwen adapter still trains on the GPU. For the whole run a second thread keeps an ANEForge linear Adam step and two 512-by-2048 matmuls resident in one program, and the run stops if either program's device mask is not `0x4`. A 256-wide matmul on this machine is dispatch-floor bound at a predicted 63 us. The wider program compiled to `0x4` and a 20-call timing was 1.29 ms per call. That engine graph is not the Qwen adapter. The three-epoch result is not in yet.

## 2026-10-28 — LoRA branch on the Neural Engine

The side matmul is no longer the GPU+ANE training path. `placement.mode: gpu_ane` keeps the frozen 4-bit Qwen base on the GPU. The trainable product `(x @ A) @ B` and its gradients are ANEForge programs on device mask `0x4`. A mask other than `0x4` raises. On a 16-by-4-by-12 check, the engine gradients matched the MLX gradients at cosine 0.998 or better. `mx.compile` cannot call the engine, so that path runs the step without it. The launch log is `run.log` inside the timestamped run directory. The shared `run.log` is not the live log. The three-epoch result is not in yet.

## 2026-10-28 — Decoder block on the Neural Engine

The 20:45:53 launch died in the first training step with `AttributeError: 'Dropout' object has no attribute 'p'`. MLX stores the keep probability on `_p_1`. No training report was written. The rank-8 product path is not the GPU+ANE step anymore. Each such call is under the dispatch floor and it flushes the GPU graph.

`placement.mode: gpu_ane` now keeps blocks 0–26 as 4-bit QLoRA on the GPU. Block 27, the frozen base plus its LoRA, is one ANEForge program on device mask `0x4`. The host pads that program to sequence 512. Measured on this M4: one block at sequence 512 compiled in 2.8 s with no dispatch-floor warning, and a separate call timed the forward at 4.9 ms and the backward at 12.2 ms, with finite gradients. Two blocks produced a non-finite backward. Four blocks compiled a backward that failed to execute. Eight blocks failed in `ANECCompile`. A later one-step loss on random tokens was finite, and the LoRA gradients on block 26 (GPU) and block 27 (Neural Engine) were finite. A one-block toy matched the MLX block output at cosine above 0.98, and the output-projection gradient matched a finite difference within 25%. The devices alternate inside the step. The three-epoch result is not in yet.

## 2026-10-28 — Neural Engine backward loss scale

The 21:16 launch compiled block 27 and trained. Validation loss before any update was 4.191, perplexity 66.11. Train loss then rose from 4.56 at step 10 to 8.87 at step 50 and was NaN from step 60. The process was still stepping and was stopped around step 300. There was no traceback. The run directory is `evidence/qwen3-0.6b-curve/2026-10-28T21-16-00`.

On one WikiText batch the block forward matched MLX, loss 3.957 against 3.963. The language-model cotangent peaked near 1e-3. At that magnitude the fp16 backward for the query factor had cosine about -0.08 against the float32 gradient. Scaling the cotangent so its peak is 1, then dividing the gradients on the host, brought the hidden cotangent and every LoRA factor to cosine 1.000. A peak near 16 overflowed the same block. The host now applies that scale, and a non-finite training loss stops the run. The three-epoch result is not in yet.

## 2026-10-28 — Backward peak backs off

The 21:38 launch matched the GPU-only losses through step 60, where the window loss was 4.84. The next step raised `neural engine backward is not finite`. No checkpoint was written. Peak 1 overflows on that batch. The backward now retries cotangent peaks 0.25, 0.0625, and 0.015625, and records the first peak below 1. A 70-step replay with the same seed and batch 2 passed that step at peak 0.25. The step loss was 10.38, the next step was 3.62, and the mean of steps 63–70 was 4.14. The three-epoch result is not in yet.

## 2026-10-28 — Recipe, not the RMSNorm backward

The 22:12 launch died at step 111 with `neural engine cotangent is not finite`. A replay of that old recipe (LoRA scale 20, constant AdamW 1e-4, no clip) reproduced it. At the failing state the float32 block output was already 105984, past the fp16 maximum, so the engine forward returned inf and the head cotangent became NaN. Steps 91–100 of that launch included one 77 s step; replays of the same code did not repeat it.

An isolated aneforge `rms_norm` backward on a 512-by-1024 input with outliers of 4608 matched a float64 reference at cosine 1.00000. That overflow hypothesis is not what happened. On GPU-trajectory states, the block-27 backward at cotangent peak 1 was finite, with the hidden cotangent and every LoRA factor at cosine at least 0.9957 against float32. A float32 block 27, with no Neural Engine at all, diverged under the old recipe (10-step means 7.8–11 from step 120 to 200). The old recipe is the measured cause. The fp16 overflows were the consequence.

The training config now follows the published QLoRA and Qwen3 LoRA recipes: rank 8, mlx-lm scale 2.0, dropout 0, AdamW peak 1e-4, weight decay 0.01, 300 warmup steps from 1e-7, cosine decay to 1e-6, global gradient clip 1.0. Those values are read from `configs/qlora_m4.yaml`. Sources: mlx-lm discussion 1985, arXiv 2106.09685, arXiv 2305.14314, the Unsloth LoRA hyperparameter guide, and LLaMA-Factory `examples/train_lora/qwen3_lora_sft.yaml`. Scale 20 at 1e-4 was about ten times mlx-lm's own default pairing of scale 20 with 1e-5.

A 260-step GPU plus Neural Engine replay, seed 0, batch 2, real WikiText order, stayed finite. Ten-step means were 5.14, 4.11, 4.50, 4.62, 3.95, 3.98, 3.52, 3.70, 3.59, 3.45 at steps 10–100 and 3.08, 3.10, 3.23 at steps 200, 210, and 260. Step 61 was 3.87 and step 111 was 3.23. All 520 row-backwards used peak 1. Forward cosine against the GPU block stayed at least 0.99996. The hidden cotangent and every LoRA B factor stayed at cosine at least 0.9986. LoRA A cosines were low in the warmup (0 at steps 1–2, 0.40 at step 4) because B is still near zero and the A gradient is below fp16; the norm ratio on those factors was about 0.02–0.4, and the B gradients that actually move the adapter matched. The three-epoch result is not in yet.

## 2026-10-29T17:02:34.803510+00:00 — Intake S158

Added [S158](sources/S158.md) as unread; source synthesis is pending.

## 2026-10-29T17:02:34.826336+00:00 — Intake S159

Added [S159](sources/S159.md) as unread; source synthesis is pending.

## 2026-10-29T17:02:34.848237+00:00 — Intake S160

Added [S160](sources/S160.md) as unread; source synthesis is pending.

## 2026-10-29T17:02:34.870700+00:00 — Intake S161

Added [S161](sources/S161.md) as unread; source synthesis is pending.

## 2026-10-29 — ANEForge pages are searchable

The earlier crawl had saved S38–S157 as HTML and left them out of `raw/manifest.json`, so they were not sources and `wiki.py search` could not see them. S38–S125 are now registered. Each of those cards carries a short extract of the rendered page so a search hits the page that owns the term. The call rules are in [Write an ANEForge program](guides/aneforge.md) and the symptom table is in [ANEForge lookup](queries/aneforge.md).

S126–S157 were GitHub directory pages (fonts, badges, workflows, folder listings). They repeated the file tree in [S41](sources/S41.md) and were removed, notes and snapshots together. They had never been manifest records.

Four files from commit `896c4943c5ebf811c0e815f8d1381fd1e157fa7d` were read and saved: [S158](sources/S158.md), [S159](sources/S159.md), [S160](sources/S160.md), [S161](sources/S161.md). The paper PDF [S38](sources/S38.md) is still not text-extracted. The architecture-guide chapters were extracted into their cards and were not rewritten as call rules, except the sentences cited in the guide.

## 2026-11-10 — Stage bounded M1 single-machine experiments

Created `research/m1-8gb-single-machine` from local main `d67f2e3` after the
sandbox denied writing Git metadata; remote freshness was not checked. Added
machine-specific GPU and hybrid smoke profiles, shared read-only preflight,
and checks before run allocation for profiles with a safety section. Explicit
MLX budgets now bound the process wired allowance and cache as well as MLX
allocations. These are software limits, not a bound on total process or ANE
compiler memory. No system memory/security settings changed.

**Inspected scope:** shared training/memory runners, Qwen3 suffix implementation,
federated ANE rejection, and host metadata. `sysctl` memory retrieval was denied;
selected `system_profiler` fields recovered 8 GiB RAM and Apple M1 without
retaining serial identifiers. [Preflight measurements](experiments/2026-11-10-m1-preflight.md)
blocked both arms on macOS 13.4.1, absent accelerator/telemetry packages, and a
proposed disk reserve. No accelerator or research training ran. ANE-only Qwen3
and separate-client ANE training remain unavailable, not GPU fallbacks.

The MLX installation guide's OS requirement and wired/cache API pages were
inspected online as implementation references, not ingested as source cards.
The wired API is documented as useful on macOS 15+. An ANEForge PyPI open failed
with a cache miss; no new compatibility claim relies on that failed retrieval.
Existing source snapshots and historical evidence were untouched.

**Validation:** 109 tests passed, 5 opt-in hardware tests skipped; one inherited
SciPy/NumPy compatibility warning. The required suite first hit sandbox socket
restrictions and missing dependencies, then passed with localhost access and
small dependencies installed only in the project environment. Code/test Ruff
passed. Whole-tree Ruff found 13 existing violations in immutable snapshots.
[Validation record](../evidence/m1-preflight-2026-11-10/validation.txt).
The [workflow](guides/single-machine.md) and [project state](project/state.md)
describe the scope. Structural lint is recorded after this appended entry;
passing it does not validate hardware, facts, or novelty.

## 2026-11-10 — M1 GPU verified after macOS upgrade

**Measured follow-up:** macOS now reports 27.2.1; disk space is about 28 GiB,
clearing the OS and proposed disk-reserve gates. Installed MLX and mlx-metal
0.32.4 into the existing project virtual environment only. An explicit GPU
float32 16×16 matrix probe returned 4096.0. Both existing opt-in MLX/CPU parity
tests passed (forward and one SGD step on two small architectures), no skips,
with unchanged tolerances. [Report](experiments/2026-11-10-m1-mlx.md),
[raw evidence](../evidence/m1-mlx-2026-11-10/README.md).

The commands ran outside the restricted sandbox with Metal access. No model
or dataset was downloaded; no QLoRA, ANE, throughput, or power result exists.
GPU QLoRA still needs mlx-lm and mactop; hybrid also needs ANEForge. ANE-only
Qwen3 remains unavailable. Preserved the earlier preflight evidence and added
new reports instead. Updated state, index, machine README and relevant guides.
Probe Ruff and structural wiki lint pass; these checks do not establish
large-model fit or ANE correctness.

## 2026-11-10 — M1 training pilots; real-model ANE gate failed

**Measured:** GPU and hybrid Qwen3-0.6B 4-bit pilots each completed 10 updates,
batch 1, sequence 128, 32 training records, seed 0, checkpointing, two fixed
evaluation batches. Both processed 1,086 training tokens. Sampled test
perplexity: 97.5221 to 73.5956 (GPU), 97.5221 to 112.5949 (hybrid). The hybrid
is rejected as correct training: three real batches gave forward cosine
0.914–0.920 and zero LoRA gradients where GPU gradients were nonzero. The
ANE suffix equaled its input; isolated input RMSNorm returned zeros. Input
peaks 5,760–6,624. Overflow is a hypothesis; this does not revise M4 evidence.
Tiny ANE linear/SGD/100-update checks and the tiny decoder finite difference
passed. [Report](experiments/2026-11-10-m1-training.md),
[raw evidence](../evidence/m1-training-2026-11-10/README.md).

**Corrections before comparison:** the WikiText helper had recorded a pinned
revision while requesting unversioned dataset-server rows. It now downloads
the immutable parquet revision and verifies the three source SHA-256 values.
PyArrow was added to the serving extra and lockfile. Training and evaluation
RNG are isolated from setup and each other; tests cover repeatability and
resume batch order. Earlier uncontrolled pilots remain retained, superseded
for comparison. No old source snapshots or historical evidence were edited.

**Resources and authorization:** first GPU attempt stopped above 512 MiB swap
growth; the user saved work and closed heavy apps before retry. First hybrid
attempt stopped below 15 GiB disk; the user explicitly selected 10 GiB for
later smoke runs. Memory/swap/time guards stayed unchanged. No unrelated
apps/files were closed/deleted. ANEForge initially used its default cache
(~139 MiB); later cache writes go under ignored runs. The precise shared-volume
disk change is not attributed to this experiment. No system security/memory/
fan settings changed. Telemetry zeros on ANE/CPU/DRAM are unvalidated for
comparison; accepted energy is unavailable. Program masks were 0x4.

**Stop decision:** the M1 hybrid config requires a passing real-block gradient
report and is now blocked. No longer hybrid or three-epoch run was launched.
The independent full-Qwen3 ANE client still rejects execution, so split-client
training was not substituted with a CPU or linear workload. Next prerequisite
is correct real-model normalization/backward, followed by a full ANE executor.

**Retrieval/validation scope:** the web fetch of the mactop latest release timed
out and the dataset-tree URL was inaccessible via web; explicit HTTPS requests
recovered the official release metadata/archive and pinned dataset tree. The
mactop archive SHA-256 matched release metadata before execution. Installed
ANEForge norm lowering/backward code was inspected; no private API settings
were changed. New code/tests pass Ruff; whole-tree Ruff retains 13 immutable-
source violations. Full-suite and structural checks are recorded alongside
the experiment; they do not establish scientific correctness or novelty.

## 2026-11-10 — Full GPU baseline launched; standalone user-run harness

The user explicitly selected three full epochs, batch 1, checkpointing off,
over the whole training split (23,767 nonempty records). The model retains the
512-token cap, so long records are truncated. This is an M1 local baseline,
not the proposal's 4,096/batch8/accum2 FedAvg recipe. A ten-step full-split fit
and three maximum-length updates completed. Peaks: 1,194,827,554 and
2,185,500,166 bytes. The latter exceeds the nominal 2 GiB allocator setting
slightly; no strict total-process bound is claimed. The initial maximum-length
diagnostic had a dataset-wrapper interface error, fixed before the measured run.
The full supervised job launched at 04:44:55 America/New_York; initial test
evaluation completed and initial validation was in progress at inspection.
No final result exists. [Record](experiments/2026-11-10-full-gpu-baseline.md).

Added nullable full-split handling through main/batch fitting, a separate
baseline admission profile (removing smoke-only step/checkpoint restrictions),
30-second telemetry, and supervisor interruption/low-battery handling. Resource
guards remain; a 48-hour ceiling allows the requested sustained run. No ANE
gate was bypassed, no other apps were closed, and no system settings changed.

At the user's request, [the standalone runner](guides/baseline-suite.md) plans
without training by default, runs supported local arms sequentially, prints
progress, refuses overlapping supervised jobs, and saves JSON/CSV summaries.
It can import a completed matching GPU run instead of repeating it. Exact
proposal data/batch settings and the full-data M1 recipe are distinct scopes.
FedAvg, all-reduce, combined/split clients, topology/payload/FedNova variants,
and multi-Mac requirements remain explicit coverage gaps; local curves are
not relabeled as those methods. Current ANE correctness failures remain gated.

Validation: 120 passed, 5 hardware tests skipped; project-code Ruff and wiki
structural checks pass. Whole-tree Ruff retains 13 immutable-source violations.
The inherited SciPy/NumPy pair emits one warning. A launched job and a planned
suite are not completed research results.

## 2026-11-10 — User-selected switch to the 4,096-record comparison

The user reported that M4 runs likely use about 4K records, batch 1, and
checkpointing false, then explicitly selected stopping the full-data job and
starting a fresh 4,096-record comparison. Verified supervisor identity before
signaling only that job. Its exit record reports `interrupted`, elapsed
1,340.24 seconds; last periodic training report 800, no checkpoint yet. All
existing artifacts were retained. No unsaved state was resumed or relabeled
as a completed full-data result.

The new supervised GPU job launched 05:08:41 America/New_York, with 4,096
records, three epochs, batch 1, accumulation 1, checkpointing false, a 512-token
cap and full evaluation. Planned updates 12,288; checkpoints every epoch.
Other optimizer/model settings and resource guards remain. Initial evaluation
was progressing with normal pressure and AC power at inspection. The actual
M4 config has not been inspected, so matching remains provisional. This is
local AdamW, not a measured FedAvg baseline. [Report](experiments/2026-11-10-m1-comparison-switch.md).

The standalone runner now defaults to `comparison`, with `full` and exact
`proposal` scopes retained explicitly. Added a recipe regression check.
Validation: 121 passed, 5 skipped; project-code Ruff, wiki structural lint and
whitespace checks pass. No scientific completion or exact M4 equivalence is
claimed. Earlier evidence snapshots and personal files were not modified.

## 2026-11-10 — Comparison resource stop and configurable swap policy

The independent supervisor stopped the comparison at progress-bar update
3,131/12,288 after 3,015.81 seconds. Last training report: update 3,100. System
swap rose by 636,354,560 bytes from launch; the 512 MiB growth guard fired.
Pressure was Warning with 991,772,672 bytes available; no OOM exception was
observed. System telemetry cannot attribute all growth to this job. The first
checkpoint had been scheduled at update 4,096, so no resumable state exists.
[Raw record](../evidence/m1-comparison-stop-2026-11-10/README.md).

In response to the user's question, added positive-integer
`safety.max_swap_growth_bytes`, preserving the prior 512 MiB default and other
guards. The supervisor records its resolved policy per attempt. Tests cover
custom thresholds, boundary behavior, preexisting swap, and invalid policies.
Changed future comparison state saves to every 500 updates; gradient
checkpointing remains false. Model/optimizer saves do not change effective
batch or optimizer math, but their I/O cadence matters for wall-time comparison.
No old artifacts were rewritten, no limit was raised automatically, and no
new training job was launched. Updated status and standalone restart guidance
to avoid importing the incomplete run as if it had finished.

Validation of this change: 128 passed, 5 hardware tests skipped; project-code
Ruff, wiki lint and whitespace checks passed. The inherited SciPy/NumPy warning
remains. No larger swap allowance has been tested in a hardware run.

## 2026-11-10 — Diagnose unchanged retry policy; expose a CLI override

The user-run comparison-retry-1 reached update 232, then stopped after 963.06
seconds. Both its copied config and guard-policy record still used 512 MiB;
the previously documented 1 GiB example had not been applied. System swap rose
by 710,475,776 bytes; pressure was Warning with 1,388,331,008 bytes available.
No OOM exception or saved checkpoint was observed. Last periodic training
report: update 200, loss 3.4528. [Record](../evidence/m1-comparison-retry-stop-2026-11-10/README.md).

Added `--max-swap-growth-mib` to remove ambiguity about which config to edit.
The runner prints the selected allowance and persists the override in its
plan/per-arm configs. A read-only plan verifies 1024 MiB without changing the
training recipe or other guards. Default stays 512 MiB; no restart or larger-
allowance hardware run was initiated. Stopped runs now retain partial history
metrics in JSON/CSV/console output without claiming final quality. Original
run artifacts and summaries were preserved rather than rewritten.

Validation: 130 passed, 5 skipped; project-code Ruff, wiki lint and whitespace
checks passed. The inherited SciPy/NumPy compatibility warning remains.

## 2026-11-10 — User-requested removal of swap-growth stop

The user explicitly requested removing the guard that had stopped the GPU
retries. Made `safety.max_swap_growth_bytes: null` disable that specific check,
and set the comparison profile to null. Omitting the field still preserves
the older 512 MiB supervisor default in other profiles. Added the explicit
`--no-swap-growth-guard` CLI flag, mutually exclusive with a numeric allowance.
The runner prints the disabled policy; the supervisor records null and an
enabled/disabled boolean. All other checks and the 2 GiB MLX allocator setting
remain unchanged. Periodic model/optimizer saves remain every 500 updates.

Tests cover disabled behavior at arbitrarily large swap growth, omitted-field
defaults, retained recipe/other policies, and contradictory CLI/API overrides.
A read-only plan verifies the selected config. No new training was launched,
no kernel/system settings changed, and historical run evidence was preserved.
Updated user-run instructions. An OOM is not guaranteed to be a clean, isolated
Python cancellation; the final user response links the Python exception and
Apple memory/swap documentation inspected as implementation references.

Validation: 131 passed, 5 hardware tests skipped; project-code Ruff, structural
wiki lint and whitespace checks pass. The inherited SciPy/NumPy warning remains.
No test establishes that an unguarded hardware run will fit or complete.

## 2026-11-10 — Non-overwriting reruns and checkpoint resume

The user's repeated output name raised an uncaught FileExistsError before
training. Inspection found `comparison-no-swap-1` marked interrupted, not
running, with a complete update-2,000 checkpoint and a last periodic report
at 2,100. Verified that the adapter/optimizer/state files exist and that the
saved data/model/training config matches the current comparison recipe.
No inference was made about unsaved progress beyond the logged reports.

The runner now allocates a distinct sibling output directory atomically on
collision, preserving original files, and prints/persists the actual path.
Added `--resume-gpu CHECKPOINT_DIRECTORY`, mutually exclusive with importing a
completed run. Resume requires matching recipe sections and complete checkpoint
files, then delegates to the trainer's identity/restore checks. Segment timing
is labeled as such rather than presented as the full resumed experiment time.
Other filesystem errors get a concise CLI error. Tests cover collisions,
preserved sentinel files, incomplete/mismatched checkpoints, and the existing
tiny adapter/AdamW restoration invariant. No large training or resume was
launched; the user-run command is in the standalone guide.

Validation: 133 passed, 5 skipped in the default suite. The separately enabled
tiny MLX adapter/AdamW recovery test passed (1 test, 1.11 seconds). Project-code
Ruff, structural wiki lint and whitespace checks pass. The real Qwen3 checkpoint
was inspected for completeness/recipe match, not restored in a new hardware
run. [Inspection and validation record](../evidence/m1-checkpoint-recovery-2026-11-10/README.md).


## 2026-11-10 — Correct real Qwen optimizer recovery

The user's `comparison-resume-1` reached checkpoint restoration then failed at
its first update with `KeyError: 'embed_tokens'`. Inspected the actual traceback,
restore function and installed MLX optimizer initialization: tensor flattening
omits frozen-layer empty containers, and setting `_initialized` directly skipped
rebuilding them. Replaced that private flag assignment with public initialization
against trainable parameters, preserving loaded moments/step. Extended hardware
regression to frozen nested modules and a compiled next update matching
uninterrupted AdamW. The earlier linear-only validation missed this failure.

Measured a bounded real-Qwen normal-trainer resume from checkpoint 2,000 through
update 2,001: loss 2.6632693, source hashes unchanged. No long run or full evaluation
was started. [New evidence](../evidence/m1-resume-fix-2026-11-10/README.md) extends,
without rewriting, the prior inspection-only evidence. Updated standalone retry
instructions, project state and index. Validation: default suite 133 passed,
6 skipped; Metal checkpoint tests 8 passed; project code/probe Ruff clean.
Whole-tree Ruff retains 13 preexisting immutable-source errors. Full-suite retry
with loopback access passed after sandbox socket restrictions. Wiki lint checks
structure; this one-update recovery establishes neither full-run success nor
ANE correctness nor a hardware-comparison result.

## 2026-11-10 — Launch the requested GPU resume

At the user's request, ran the standalone comparison command with `--resume-gpu`
pointing to checkpoint 2,000 and `--no-swap-growth-guard`. No supervised trainer
was active before launch; free disk was 15.72 GiB. `comparison-resume-2` passed
preflight, completed initial evaluation and advanced beyond step 2,054. Left it
running independently of Codex (launcher PID 43583). No training code changed.
The latest user-reported error was unspecified and did not reproduce here.
[Observation](../evidence/m1-resume-launch-2026-11-10/README.md) records exact scope
and log paths; project state and index link it. This confirms launch and resumed
updates only, not final perplexity, all epochs, or proposal/ANE coverage.

## 2026-11-10 — Repair and validate hybrid numerics; launch comparison

At the user's request to proceed with ANE+GPU, inspected the installed ANEForge
RMSNorm emitter/VJP, real activation failures and suffix bridge. A bounded
scaling sweep fixed the large-input forward but exposed zero-input backward
errors. Added an explicitly pinned local forward lowering that preserves the
original backward coordinates/epsilon; retained the backend epsilon-floor
approximation. Normalization alone still failed small adapter gradients.
Equivalent A/16, B*16 upload plus the gradient chain rule passed those cases.
No installed dependency or historical evidence was modified.

Real-block gates passed at 128 and 512, with full-length and padded inputs,
trained adapters and gradients to the GPU prefix. A short-heading boundary
attempt failed in the MLX reference VJP; the successful longer padded retry is
reported separately. A 20-update supervised hybrid pilot exited 0 with sampled
PPL 117.18 -> 59.51, 24 ANE forwards / 20 backwards and peak MLX 1.846 GB.
[Report and raw results](../evidence/m1-hybrid-validation-2026-11-10/README.md)
extend and supersede the earlier failed implementation's launch blocker.

Wired the validated report into the runner and require matching sequence caps.
Added a matching M1 hybrid machine config; full default tests 134 passed,
7 skipped; separate numerical/decoder device tests 6 passed. Project Ruff and
whitespace checks pass; whole-tree Ruff retains 13 preexisting immutable-source
errors. Updated the standalone/ANE guides, index and state. Structural wiki lint
is separate from the numerical evidence.

Launched `comparison-hybrid-1` at 17:33:30 America/New_York, PID 57790, with
`--existing-gpu` reusing the successful GPU comparison and a fresh hybrid arm.
Observed preflight and initial evaluation; the long run is not yet a result.
User-selected disabled swap-growth stop and other configured guards remain.
Standalone ANE, federated and multi-Mac coverage are still blocked. No M4 or
energy-efficiency claim follows from these M1 measurements.

## 2026-11-10 — Resume hybrid with a 5 GiB disk reserve

The hybrid stopped near update 607 at the configured 10 GiB disk floor; checkpoint
500 contains adapter, optimizer and state files. At the user's request, added
`--min-free-disk-gib` and `--resume-gpu-ane` to the standalone runner and launched
`comparison-hybrid-resume-1` with 5 GiB, reusing the completed GPU result. Resume
checks placement mode as well as recipe; original outputs are preserved. Launcher
PID 61013 at 18:04:50 America/New_York. The actual saved guard policy confirms
5368709120 bytes. Initial evaluation was observed advancing; resumed training
completion is unknown. Log: `runs/baseline-suites/launch-hybrid-resume-20261110T180450/stdout.log`.

The existing default disk threshold is unchanged; this invocation explicitly
overrides it. Memory-pressure checks remain and swap-growth stop remains disabled.
Updated the standalone guide and state. Default suite: 134 passed, 7 skipped;
then the newly added hybrid-routing regression and related runner tests passed
(11 tests). Project Ruff passes; whole-tree Ruff has the existing 13 errors in
immutable raw source snapshots. No hardware backend code changed in this turn.

## 2026-11-10 — Retry hybrid after the user closed memory-heavy apps

User requested another restart after closing other apps. Confirmed no active
supervised trainer, complete checkpoint 500 and 13.05 GiB free disk. Started
`comparison-hybrid-resume-2`, launcher PID 71325 at 22:28:06 America/New_York,
with the same 5 GiB reserve and memory checks. Initial evaluation completed;
observed actual resumed training past 578 and normal system memory pressure,
beyond the prior retry's stop at 501. Left the job running independently.
[Observation](../evidence/m1-hybrid-retry-2026-11-10/README.md) records scope and
log paths. No code/config changes and no final training result claimed.


## 2026-11-16 — Audit completed M1 comparison and prepare baseline PR

Inspected both original result/config/history records, supervisor exits, and final
checkpoint metadata. Both local arms completed 12,288 updates with full evaluation;
GPU test PPL 23.323357, hybrid 23.323607. Copied selected raw artifacts without
modification into evidence/M1_8GB and recorded SHA-256 provenance. Prior evidence
and log entries remain unchanged. Updated state/index/baseline guide with the
[completion report](../evidence/M1_8GB/README.md).

Read PR #1 metadata/file listing, pinned M4 GPU/hybrid configs and its evidence
README at head 30471852690b55476b25f8dafb7f8b6635ba7c4e. Initial web retrieval
failed (cache miss) and sandbox curl failed DNS; direct GitHub API retrieval with
network permission succeeded. Core recipes match, but changed numerics, RNG/data
handling, resource settings and resumed segments limit cross-Mac claims. No
M4 raw telemetry audit or new M4 execution was performed. Local ANE mask/call
counts are retained; positive power/compute-plan evidence remains unestablished.
The complete federated proposal is not claimed complete.

Validation: 135 default tests passed, 7 skipped; 14 enabled hardware/checkpoint
tests passed. Maintained code Ruff passed; whole-tree Ruff retains 13 preexisting
errors in immutable raw source snapshots. No long training rerun was needed.
