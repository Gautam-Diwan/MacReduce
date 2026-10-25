# MLX validation on this Mac

**Measured locally** on 2026-10-24. Scope: one Apple M4 MacBook Air (`Mac16,12`), macOS 26.5, arm64, Python 3.12.12, MLX 0.32.4, PyTorch 2.14.1, NumPy 2.5.4. The run uses the default one-block float32 model (29,888 parameters, UTF-8 byte tokenizer, context 32). Transport is HTTP on loopback. One physical machine. This is not a Wi-Fi, multi-Mac, ANE, or M5 result.

`python3.12 -m venv` failed for the uv-managed interpreter (`stdlib dir = '/install'`). The environment was created with `uv venv .venv --python 3.12` and installed with `uv pip install --python .venv/bin/python -e '.[cpu,mlx,dev]'`.

## Commands and results

| Check | Result | Artifact |
|---|---|---|
| `macreduce doctor --smoke` | Metal matmul probe **available**, sum `4096.0` | [doctor-smoke.json](doctor-smoke.json) |
| `MACREDUCE_TEST_MLX=1 pytest -q tests/test_mlx_optional.py` | **2 passed**. Default model and a 2-layer config: forward loss and one SGD step versus CPU | [mlx-test.txt](mlx-test.txt) |
| `macreduce demo --backend mlx-gpu --rounds 3 --steps 4` | **done** in 0.817 s. Validation loss 6.058713912963867 to 3.713829755783081. Workers `small` batch 2 and `large` batch 4 | [mlx-gpu-demo.txt](mlx-gpu-demo.txt), [run directory](mlx-gpu-demo/) |
| `python -m pytest -q` without the opt-in flag | **33 passed, 2 skipped** | [pytest.txt](pytest.txt) |
| `ruff check .` | All checks passed | [ruff.txt](ruff.txt) |

Versions: [versions.txt](versions.txt). `xctrace` was unavailable (`xcrun` could not find the utility), so there is no Metal capture. The extended gradient audit and a 100-step stability run from the validation guide were not executed. The short demo shows a finite decreasing loss on `mlx-gpu`; it is not a paired CPU trajectory for those three rounds.

Chip presence and the doctor probe are separate from ANE training, which remains unimplemented.
