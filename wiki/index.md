---
title: "MacReduce knowledge index"
type: index
status: active
updated: 2026-10-23
sources: []
---

# MacReduce knowledge index

MacReduce is an observable distributed trainer for heterogeneous Macs, with physical Wi-Fi as the target network. The current measured result is a small CPU model trained by two processes on one Mac. ANE training, physical multi-Mac performance, and large-model inference remain unvalidated or unimplemented as detailed below.

**Start with [current state](project/state.md), then [what to build next](project/roadmap.md), and [research candidates](research/novelty.md).** For the proposal, read [the active project proposal](../docs/proposal.md). For runnable commands, use [the project README](../README.md).

This wiki contains maintained synthesis, not a claim that every linked source has been fully read. Source cards state reading scope. Follow [the maintenance workflow](how-to-use.md) when adding or changing knowledge.

## Start here

- [Using and maintaining the MacReduce wiki](how-to-use.md)
- [Current implementation and evidence](project/state.md)
- [Implementation order and completion gates](project/roadmap.md)
- [Training architecture and extension points](project/architecture.md)
- [Wiki schema and evidence rules](schema.md)
- [MacReduce wiki change log](log.md)

## Implementation guides

- [Validate the MLX path on the M4](guides/mlx-validation.md)
- [Run and characterize a physical WiFi cluster](guides/wifi.md)
- [Move beyond the synthetic Transformer](guides/model-scaling.md)
- [Schedule unequal Macs without hiding objective changes](guides/heterogeneity.md)
- [Integrate ANE training through falsifiable gates](guides/ane.md)
- [Make profiling useful for scientific claims](guides/profiling.md)
- [CPU matrix units M5 GPU paths and other resources](guides/accelerators.md)
- [Run one larger model across Macs as a separate path](guides/inference.md)
- [Scale transport and preserve recovery semantics](guides/transport.md)
- [Optimizer state DiLoCo and adapter aggregation](guides/optimizers-adapters.md)

## Research

- [Related work and limits on novelty](research/related-work.md)
- [Research contribution candidates and falsification plans](research/novelty.md)
- [Evaluation plan and acceptance gates](research/experiments.md)
- [Unknowns contradictions and next literature searches](research/gaps.md)

## Decisions

- [Decision 0001 Trainer first over Wi-Fi](decisions/0001-scope.md)
- [Decision 0002 Preserve measurement and objective semantics](decisions/0002-measurements.md)

## Templates

- [Experiment report template](templates/experiment.md)
- [Source synthesis template](templates/source.md)
- [Decision record template](templates/decision.md)

## Sources

- [S01 — Karpathy LLM Wiki](sources/S01.md)
- [S02 — MLX distributed communication](sources/S02.md)
- [S03 — AirTrain](sources/S03.md)
- [S04 — DiLoCo](sources/S04.md)
- [S05 — Streaming DiLoCo](sources/S05.md)
- [S06 — FedNova](sources/S06.md)
- [S07 — AdaComm](sources/S07.md)
- [S08 — HEW Local SGD](sources/S08.md)
- [S09 — FusionML CPU GPU coexecution](sources/S09.md)
- [S10 — ANEForge training](sources/S10.md)
- [S11 — maderix ANE training](sources/S11.md)
- [S12 — Orion](sources/S12.md)
- [S13 — M5 GPU neural accelerators](sources/S13.md)
- [S14 — Apple XNU SME](sources/S14.md)
- [S15 — MLX Metal capture](sources/S15.md)
- [S16 — Core ML profiling](sources/S16.md)
- [S17 — mactop](sources/S17.md)
- [S18 — EXO](sources/S18.md)
- [S19 — MLX LM LoRA](sources/S19.md)
- [S20 — FedAvg](sources/S20.md)
- [S21 — Apple M1 announcement](sources/S21.md)
- [S22 — Hello SME](sources/S22.md)
