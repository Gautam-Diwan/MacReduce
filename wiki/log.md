---
title: "MacReduce wiki change log"
type: log
status: active
updated: 2026-10-24
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
