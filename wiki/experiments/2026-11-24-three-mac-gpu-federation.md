---
title: "Three-Mac GPU Flower/Tailscale connectivity pilot"
type: experiment
status: measured
updated: 2026-11-25
sources: ["S166", "S175", "S181", "S182", "S183"]
---

# Three-Mac GPU Flower/Tailscale connectivity pilot

## Question and preregistered hypothesis

Can the three reported Macs complete a fixed GPU FedAvg cohort over Tailscale using the pinned Qwen3-0.6B-4bit LoRA trainer, with every expected logical ID present and no partial average?

This was a connectivity/trainer smoke test, not the proposal's three-epoch perplexity experiment. Falsified if any client is missing, a delta is nonfinite, identities disagree, or the coordinator commits fewer than three rounds. No speedup or quality hypothesis was pre-registered.

## Reproduction

Inspected artifacts on the M4 Air checkout at git `92b3cff`, run `ce017ce1-1a68-489a-bc09-a8bcdd299cc2`, 2026-11-24 evening America/New_York. Commands and compact receipts: [evidence/tailscale-federation-2026-11-24](../../evidence/tailscale-federation-2026-11-24/README.md). Config [configs/federation.yaml](../../configs/federation.yaml): `mode: gpu`, `coordinator: m4-air`, microbatch 1, two local steps, three rounds, constant 1e-4, rank 8 / scale 2, WikiText-2 prefix 4,096, sequence 512. HMAC token was loaded from local `.env` and is not recorded.

## Raw artifacts and validity checks

Coordinator `state.json` and three `worker.json` files share `run_id`, `plan_hash`, and `data_hash`. Final `adapter-000003.npz` SHA-256 matches the published pointer. Air `delta.npz` hashes match `delta_file_hash`. Flower reported 3/3 results and 0 failures for each round. Air GPU preflight had passed before join. M1 and Pro worker trees were copied later; their filesystem mtimes are not training clocks. Overlay IPs are redacted in retained terminals. No `tailscale ping` / `netcheck` capture. No mactop.

## Results

**Measured:** three physical GPU clients finished the pilot. Six optimizer updates per client. 5,046,272 LoRA parameters, ~20.3 MiB fp32 payload, all finite. M1 with `grad_checkpoint: true` completed these six steps. Round barriers 61 s, 55 s, 87 s after the Pro joined. About twenty minutes of wait before the first commit. Air local compute 1–4 s per round. Total supervised tokens across clients: 1,931. Evaluation skipped.

**Inferred, not measured:** most barrier time is M1 and/or transport, not Air GPU. Scaling K=2 to a 12,288-step curve would be tens of hours of round overhead; that is an extrapolation, not a timed run.

**Unknown:** direct vs relayed path, energy, M1 three-epoch memory, `gpu_ane`.

## Interpretation and next step

The November 12 loopback/preflight record is superseded for the claim "no physical three-Mac run." It is not superseded as the software-integration record. Calibration, full eval, and combined/split arms remain [roadmap](../project/roadmap.md) TODOs. Next useful timing run: one K=32 round with per-worker start/end logs and a Tailscale path capture.
