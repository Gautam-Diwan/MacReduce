# Local validation evidence

Collected October 22, 2026 on an Apple M4 MacBook Air, 24 GiB RAM, macOS 26.5, Python 3.12.12. The source package includes exact dependency versions in `requirements-tested.txt`.

| Check | Observed result |
|---|---|
| Automated suite | **20 passed, 1 skipped.** The skipped check explicitly requires Metal hardware access. |
| End-to-end reference run | Two separate worker processes, real HTTP over loopback, one physical Mac. |
| Model | 29,888-parameter causal byte Transformer; float32; 32-token sequences. |
| Training | Eight rounds, ten local SGD steps per worker per round, batches 2 and 4, learning rate 0.05. |
| Processed tokens | 15,360 committed training tokens. |
| Validation loss | **6.058714 → 1.768063**, fixed synthetic validation windows. |
| Checkpoint payload | 123,204 bytes per model snapshot. |
| CPU operator profiling | Trace contains 1,484 total events and 97 distinct CPU-operator names/categories, including matrix multiplication and backward operations. |
| Accelerate probe | 512×512 float32 SGEMM, three warmups, ten measurements; sampled numerical check passed. Internal matrix-unit dispatch remains unverified. |
| MLX GPU smoke test | Import failed because this sandbox cannot access a Metal device. |
| mactop | v2.1.5 installed; collection failed because the sandbox denied CPU feature queries. No power readings were obtained. |
| Instruments tooling | `xctrace` was unavailable in the installed developer tools. |

The reference run's elapsed duration is recorded in `reference-run/result.json`. It includes worker startup and polling but begins after coordinator initialization. It is not a Wi-Fi benchmark, sustained throughput measurement, or cluster speedup result. Both workers contend for the same physical CPU and memory. The small model is intentionally a transport and learning smoke test.

## Files

- [Reference run report](reference-run/report.md): phase durations, local training rates, transfer totals, and validation history.
- [Run result](reference-run/result.json): completion, losses, settings, and explicit untested fields.
- [Committed state](reference-run/state.json): model configuration, dataset fingerprint, receipts, checkpoint name and checksum.
- `reference-run/events/`: original coordinator and worker JSONL measurements.
- `reference-run/*.trace.json`: per-node application timelines. Each node has its own time origin.
- [CPU operator trace](cpu-operators.trace.json): one post-warmup step from a separate two-round instrumented demo, not the reference performance run.
- [Hardware inventory](hardware.json): detected M4 identity and unavailable probes, with reasons.
- [Accelerate result](accelerate.json): raw short-probe timing and correctness result.
- [mactop collection status](mactop/mactop.status.json): failure reason, version, and attempted command.
- [Test output](tests.txt): final suite result, including the optional hardware test skip.
- [Dependency versions](versions.json): installed package versions.

The tests cover unequal-token averaging, a one-step pooled-SGD numerical reference, causal attention, unsafe or malformed checkpoints, duplicate and conflicting updates, wrong run/version/token counts, checkpoint checksum failures, committed and uncommitted restart semantics, round deadlines, real HTTP authentication and size bounds, and a lost acknowledgement after an update has already been accepted.

The final HTTP tests and reference run passed after network permission was restored for local loopback connections. An earlier retry after resuming the task failed at socket binding because the previous turn's permission had expired; that was an environment restriction, not a training result.

Physical multi-Mac operation, Wi-Fi throughput, mixed CPU/MLX numerical parity, GPU capture, TLS connections, ANE training, and M5 neural-accelerator execution remain unverified. Missing measurements are not interpreted as zero utilization.
