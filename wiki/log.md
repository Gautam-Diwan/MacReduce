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

