---
title: "Scale transport and preserve recovery semantics"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S05"]
---

# Scale transport and preserve recovery semantics

Current checkpoints are complete uncompressed NumPy archives. The client limits responses to 2 MB, and the coordinator retains full worker updates. These are explicit prototype limits, not a production transport design.

## Streamed artifact protocol

Add a manifest with run ID, round-plan hash, model version, node, parameter schema, dtype, total length, content hash, and chunk hashes. Validate maximum size before allocation. Write incoming chunks to a temporary file, cap concurrent buffers, and atomically mark a completed artifact only after all hashes match. Reduce from validated artifacts or bounded parameter blocks. Keep parsing free of executable serialization formats.

Assign an update identity independently of chunk retries. Retransmitting a chunk or acknowledgement must not trigger another optimizer step. A client should persist the outgoing update and metadata until it receives a durable receipt. After restart, recover the pending artifact or recompute only from a documented common checkpoint and deterministic inputs. Never average partially received updates.

## Fault-injection plan

Test interrupted upload, lost final acknowledgement, corrupted chunk, incorrect final hash, duplicate/conflicting identity, stale run, disk full, process termination before and after manifest rename, and slow-client timeouts. Define which failures require a complete round replay. Document that the current atomic writes address process restart; a claim of power-loss durability needs filesystem/directory-fsync analysis and explicit tests.

## Compression order

Start with lossless compression and measure serialization CPU time, transferred bytes, and complete-round latency. Model weights may compress poorly. Then evaluate float16 or quantized deltas as an explicit algorithm change, tracking numerical error and quality at fixed tokens. Error-feedback buffers, if used, become part of durable optimizer/communication state.

[Streaming DiLoCo](../sources/S05.md) is relevant prior art for blockwise communication, overlap, and quantized exchange. Copying that idea is not a novel contribution. Before overlapping future local work with a transfer, specify exactly which base version each block represents and how stale results are applied. Compare against the uncompressed synchronized reference, not just raw network throughput.

**Acceptance:** bounded memory, validated payloads, deterministic retry behavior, and a quality/latency tradeoff measured on real Wi-Fi. See [Wi-Fi measurements](wifi.md) and [experiments](../research/experiments.md).
