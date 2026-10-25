---
title: "Scale transport and preserve recovery semantics"
type: implementation
status: active
updated: 2026-10-24
sources: ["S05"]
---

# Scale transport and preserve recovery semantics

Checkpoints remain complete uncompressed NumPy archives, now with an embedded model config. Workers download and upload those archives as chunked artifacts. A single HTTP response is capped at 1 MiB for control messages and at 256 KiB for one chunk. The whole artifact is rejected above 64 MiB before a payload buffer is allocated. The coordinator still retains each validated worker update until the round commits. Compression, blockwise deltas, and physical Wi-Fi timing are not implemented.

## Streamed artifact protocol

**Implemented.** A manifest carries schema, run ID, version, role, node, parameter schema, dtype, plan hash, total length, content hash, chunk size, and chunk hashes. Size is checked before the temp file is created, and the declared length is not preallocated. Chunks are written to a temporary file. The file is renamed to a completed artifact only after every chunk hash and the content hash match. The reducer then runs once on those bytes. Parsing stays on NumPy archives with `allow_pickle=False`.

Update identity is the run, version, and node, independent of chunk retries. Resending the same bytes returns the existing receipt and does not apply the update again. A different payload for that identity is rejected. A complete call before every chunk arrives does not enter the reducer. In-flight uploads live in process memory: a coordinator restart drops them and replays the round from the last committed checkpoint. The client rebuilds an update from that checkpoint and the deterministic local batches; it does not yet persist the outgoing bytes across its own restart.

Tests cover the happy path, a corrupted chunk, a duplicate retry, an oversize manifest, and a partial upload. They do not cover every case below.

## Fault-injection plan

Test interrupted upload, lost final acknowledgement, corrupted chunk, incorrect final hash, duplicate/conflicting identity, stale run, disk full, process termination before and after manifest rename, and slow-client timeouts. Define which failures require a complete round replay. Document that the current atomic writes address process restart; a claim of power-loss durability needs filesystem/directory-fsync analysis and explicit tests.

## Compression order

Start with lossless compression and measure serialization CPU time, transferred bytes, and complete-round latency. Model weights may compress poorly. Then evaluate float16 or quantized deltas as an explicit algorithm change, tracking numerical error and quality at fixed tokens. Error-feedback buffers, if used, become part of durable optimizer/communication state.

[Streaming DiLoCo](../sources/S05.md) is relevant prior art for blockwise communication, overlap, and quantized exchange. Copying that idea is not a novel contribution. Before overlapping future local work with a transfer, specify exactly which base version each block represents and how stale results are applied. Compare against the uncompressed synchronized reference, not just raw network throughput.

**Acceptance:** bounded memory, validated payloads, deterministic retry behavior, and a quality/latency tradeoff measured on real Wi-Fi. See [Wi-Fi measurements](wifi.md) and [experiments](../research/experiments.md).
