---
title: "Experiment report template"
type: template
status: template
updated: 2026-10-23
sources: []
---

# Experiment report template

Copy this page into `wiki/experiments/` with a descriptive filename, actual date, sources, and appropriate status. Link the completed report from the index and log. Unfilled fields below are not results.

## Question and preregistered hypothesis

State the question, comparison, primary metric, quality threshold, stopping budget, and what would falsify the hypothesis. Distinguish exploratory choices from choices fixed before the run.

## Reproduction

Record exact command, source revision/hash, dependency versions, seeds, hardware/OS, power/thermal observations, network topology, model/tokenizer/data hashes, data permissions and split, optimizer/precision, p_i/K/batches, transport settings, and profiling mode. Do not include secrets.

## Raw artifacts and validity checks

Link run manifest, logs, checkpoints, timing/byte records, traces, and test output. State numerical checks, instrument limitations, missing values, recovery events, and whether this was physical Wi-Fi, loopback, or injected conditions.

## Results

Give all run outcomes, replicate count, uncertainty, time-to-quality, quality-at-fixed-tokens, and overhead. Label source-reported comparisons separately from our measurements. Account for failed or incomplete trials.

## Interpretation and next step

State supported conclusions, alternative explanations, limitations, affected wiki pages, and the next test. Append a log entry and run wiki lint after adding this report.
