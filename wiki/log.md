---
title: "MacReduce wiki change log"
type: log
status: active
updated: 2026-10-25
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
