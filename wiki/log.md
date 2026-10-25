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
