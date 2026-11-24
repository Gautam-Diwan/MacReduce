"""
MacReduce research prototype. Hardware claims require measured evidence.

Package layout:

- ``cluster`` — retained local-SGD coordinator, workers, artifacts, toy model
- ``train`` — Qwen3 QLoRA curve, memory budget, batch search, train-plan
- ``federation`` — Flower/Tailscale transport, FedAvg identities, LoRA client
- ``ane`` — Neural Engine ranking, ANEForge programs, Qwen suffix compile
- ``serve`` — phase-placement scheduler and mlx-lm executor
- ``observe`` — host events, mactop traces, plots, Accelerate probe
"""

__version__ = "0.1.0"
