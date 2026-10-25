# MacReduce migration verification

This directory contains new results from the renamed project at its canonical course-folder location. Earlier evidence remains unchanged.

- `tests.txt`: 25 passed, one opt-in MLX hardware test skipped.
- `lint.txt`: Python static checks passed after fixing import grouping following the package rename.
- `reference-run/result.json`: two CPU worker processes, one physical Mac, loopback HTTP, eight rounds, validation loss 6.058714 to 1.768063.
- `preservation.json`: all 31 original evidence files retain their recorded SHA-256 values; the former code directory no longer exists.
- `wiki-lint.json`: wiki metadata, local link targets, index coverage, and all 22 raw source checksums.

These checks do not establish physical Wi-Fi scaling, GPU or ANE execution, large-model feasibility, or research novelty. The shared development environment used for verification lives in the original task workspace; the project itself is relocatable and README documents fresh environment setup.
