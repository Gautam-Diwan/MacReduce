# Handoff: GPU + Neural Engine QLoRA run (2026-10-28, late evening)

Written by the agent that worked on this from about 22:55 to 23:50 on 2026-10-28. It stops here because the user ran out of budget. Everything below is either **measured** (I ran it and read the numbers), **prior session** (reported by the earlier agent or the wiki, not re-verified by me), **source-reported** (external documents), or **proposed**.

No trainer is running. I did not launch one. `ps aux | rg main.py` returned nothing at 23:50. `evidence/qwen3-0.6b-curve/latest` still points at `2026-10-28T22-12-30` (dead, status error).

## 1. Goal

One GPU + ANE QLoRA run of `mlx-community/Qwen3-0.6B-4bit` on WikiText-2 (`data/wikitext-2-raw-v1-curve-4096`) that finishes 3 epochs with finite, non-diverging loss. Entry point from the MacReduce root:

```
.venv/bin/python main.py --config configs/qlora_m4.yaml
```

Package `macreduce`. Do not commit, push, email, or delete evidence. Do not claim the 3-epoch curve is measured until the run directory has `result.json` with `"status": "ok"`.

## 2. Hardware and pins

M4 MacBook Air, 24 GB. mlx 0.32.4, mlx-lm 0.32.2, aneforge 0.4.0 (pins are prior session, consistent with what I imported). Python `.venv/bin/python` (3.12). Train through `main.py`, not `macreduce lora`. One trainer at a time. Detach with

```python
subprocess.Popen([sys.executable, "main.py", "--config", "configs/qlora_m4.yaml"],
                 cwd=root, start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

macOS has no `setsid`; do not use `nohup`. Each launch creates `evidence/qwen3-0.6b-curve/<local timestamp>/` with `run.log` (tqdm, convert `\r` to `\n` when reading: `tr '\r' '\n' < run.log`), `run.pid`, `history.jsonl`, `config.yaml`, `figures/`, `checkpoints/`. The parent `evidence/qwen3-0.6b-curve/run.log` and parent `figures/` are stale and immutable; parent `history.jsonl` is the stable GPU-only reference trajectory. Check `ps` before believing a poll that says a trainer died. Kill with `kill -TERM -<pid>` on the process group and confirm with `ps`.

## 3. Locked architecture

Blocks 0–26: 4-bit QLoRA on the GPU in MLX. Block 27 (frozen base dequantized to fp16, LoRA A/B as live inputs): one ANEForge forward program and one backward program, device mask `0x4` (`_require_ane` raises otherwise), host-padded to sequence 512, batch rows run sequentially, called through an `mx.custom_function` with a vjp. Embedding, final norm, and the head stay on the GPU. Devices alternate inside a step; nothing is overlapped. Width is exactly one block: a 2-block backward overflowed, 4 blocks failed to execute, 8 blocks failed to compile (prior session). Do not go back to per-linear rank-8 calls or the side 512x2048 matmul.

Code: `src/macreduce/ane_stack.py` (`SuffixEngine`, `_block`, `_project`, `_rows`, `_run_backward`, `_candidate_scales`, `_loss_scale`, `_layer_spec`). `_layer_spec` bakes `float(proj.scale)` into the program, so the compiled block follows whatever LoRA scale the model object has (verified by reading the code; at runtime my replay asserted `engine.specs[0]["scales"] == [2.0]*7` under the new recipe and passed). aneforge VJPs: `.venv/lib/python3.12/site-packages/aneforge/autograd.py` (`_vjp_rms_norm` at ~line 238, `_vjp_bmm`, `_vjp_softmax`). Graph ops: `aneforge/graph.py`. Rope/RMSNorm reference: `aneforge/llm.py`.

The backward host path (`_run_backward`) scales the cotangent so its peak is one of `(1, 0.25, 0.0625, 0.015625)`, divides the results back on the host, and keeps the first finite result. A non-finite cotangent arriving from the GPU raises `neural engine cotangent is not finite` in `_candidate_scales`.

## 4. Failure history (run directories under `evidence/qwen3-0.6b-curve/`)

Prior session unless marked measured:

- `2026-10-28T20-45-53`: `AttributeError: 'Dropout' object has no attribute 'p'` before any training step. Per-linear path; abandoned.
- `2026-10-28T21-16-00` (pid 58773, killed): validation loss 4.191 / ppl 66.1 before any update. Train loss 4.56, 5.53, 6.22, 7.69, 8.87 at steps 10–50, NaN from step 60. Cause: fp16 backward at the raw cotangent (peak ~1e-3) was wrong (query LoRA grad cosine about −0.08 vs float32). Forward matched (loss 3.957 vs 3.963, cosine 0.99999). Scaling the cotangent to peak 1 and dividing on the host gave cosine 1.000 on every factor; peak ~16 overflowed.
- `2026-10-28T21-38-09`: died at step 61 with `neural engine backward is not finite` at peak 1. Window loss 4.84 at step 60.
- Ladder of peaks added. A 70-step replay passed step 61 at peak 0.25; that step's loss was 10.38, next step 3.62, steps 63–70 mean 4.14.
- `2026-10-28T22-12-30` (pid 83494, dead, `result.json` status error): peak 0.25 first used at step 61. 10-step means at steps 10–110: 4.40, 3.65, 4.03, 4.64, 4.18, 4.84, 4.78, 4.21, 4.04, 4.27, 5.10. Died at step 111 (traceback in its `run.log`) with `neural engine cotangent is not finite`. Steps 91–100 ran at 0.108 it/s; I read the log (**measured**): step 92 alone took 77 s (08:00 to 09:17 on the tqdm clock), the other nine steps were normal. Not reproduced in any of my replays (no step over 3.3 s in 160+ steps); most likely a concurrent process (the prior agent's 70-step replay was run around that time) or memory pressure, not our code. No checkpoint (checkpoints are per epoch, step 2048).
- GPU-only reference, parent `evidence/qwen3-0.6b-curve/history.jsonl` (**measured** by reading it): 10-step means 4.39, 3.66, 4.01, 4.58, 4.35, 4.12, 3.89, 4.03, 3.98, 4.00, 4.20, 4.48, 4.11, 4.30 at steps 10–140, then down to ~3.3–3.7 by step 830, never diverging, with the OLD recipe (scale 20, lr 1e-4, no clipping).

## 5. What I measured about the ANE numerics (new in this session)

All scripts are in `/tmp` (listed in section 9). Everything here is **measured** on this M4 tonight.

**Isolated rms_norm probe** (`/tmp/rms_probe.py`): aneforge's native `rms_norm` backward on a [512,1024] input with row std up to 1152 and outliers of 4608 was finite and matched a float64 reference at cosine 1.00000 for every amplitude tested (10, 200, 300, 1000, 4608). A hand-built scale-invariant norm (`xs = x*s`, `s = 1/128`) gave the same. So the rms_norm-VJP-overflow hypothesis from the task brief was **not confirmed** in isolation; the ANE handles x² ≫ 65504 inside that op. I did not change `_block`.

**GPU-state capture** (`/tmp/capture.py 125 1,10,61,62,100,110,112,114,115,116,118,120`, writes `/tmp/cap/step-XXX.npz` and `/tmp/cap/gpu_losses.npy`): a GPU-only AdamW replay (seed 0, batch 2, real `iterate_batches` order, old recipe) reproduced the reference history to three decimals (4.393, 3.657, 4.009, 4.583, 4.348, 4.119 …). Captured |hidden| into block 27 was 5824–6816, block output 3328–5408, cotangent peak 1e-4 to 5e-3.

**ANE backward vs float32 MLX on those batches** (`/tmp/anecheck.py 1,0.25,0.0625,0`): at peak 1 every batch (including step 61 and 115) was finite with d_hidden cosine ≥ 0.9999 and worst-factor cosine ≥ 0.9957, norm ratios 0.98–1.00. At peak 0.25: worst factor 0.948–0.999. At 0.0625: worst 0.89–0.995. At the raw cotangent (no scaling): cosines 0 to 0.8, confirming the prior finding. Per-position d_hidden (`/tmp/rowcheck.py`) matched at every position including position 0 with |h| ≈ 6000 (cosine 1.0000, ratio 1.00). So, on GPU-trajectory states, the fp16 backward at peak 1 is accurate and finite.

**Instrumented ANE replay, old recipe** (`/tmp/replay.py 130 base` and `/tmp/replay.py 70 gradcheck`, logs `/tmp/cap/base.jsonl`, `/tmp/cap/gradcheck.jsonl`): reproduced the real failures. 10-step means 4.395, 3.655, 4.026, 4.643, 4.183, 4.836, 4.780, 4.209, 4.037, 4.267, 5.100. Step 61 needed peak 0.25 on one row and its loss was 8.556. Died at step 111 with the non-finite cotangent. Per-step ANE gradients were compared against a float32 MLX block on the *same* (ANE-trajectory) state: worst cosine over all 15 outputs ≥ 0.985 at every step 2–70, norm ratios 0.85–1.06. ANE forward vs GPU block on the same state: cosine ≥ 0.99998 every step until the state itself was broken. The failure dump `/tmp/cap/fail-base-111.npz` shows the **GPU** block output on that state was already 105,984 (> fp16 max), so the ANE forward returned inf and the head cotangent became NaN. One row had h2 max 977, normed 2100, gate 1230, up 797, act 7e4, down 1.07e5 in float32: the model state had diverged; fp16 only made it visible.

**Plumbing check** (`/tmp/fake_replay.py 120 plumb`): running the bf16 GPU block through the same `mx.custom_function` path reproduced the GPU-only trajectory to three decimals for 120 steps, and one-step full-model gradients matched the plain path at cosine ≥ 0.999 (`/tmp/plumb.py`). The custom-function plumbing is exact.

**Precision sensitivity** (`/tmp/fake_replay.py 200 fp32`, `/tmp/chaos.py 200 fp32block`, `/tmp/chaos.py N noise[:layer:eps]`): with the OLD recipe,
- a float32 block 27 (no ANE at all, plain nn module or through the custom function; both give identical numbers) diverged: 10.868 at step 101, 21.4 at 116, 10-step mean 7.8–11 from step 120 to 200;
- a float32 block 27 plus a 1e-6 perturbation elsewhere also diverged (mean 7.5–8 at steps 160–200);
- bf16 GPU-only with a 1e-6 perturbation to one LoRA A (three different sites) spiked to 8–9 on short seq-33 batches and 10-step means up to 5.9, but recovered to ~3.6–3.8 by steps 200–400.
Conclusion (**measured**, interpretation **inferred**): the old recipe is marginally stable and chaotic; trajectories diverge from the reference around step 19–20 under any perturbation, and a more precise block 27 (fp32, or fp16 ANE) went into a bad basin twice out of two tries. The fp16 overflows seen in the runs were consequences of that divergence, not its cause. I did not find an ANE numerical bug to fix; the peak-1 ladder is retained as a safety net.

**New recipe replay, interrupted** (`/tmp/replay.py 260 recipe`, log `/tmp/cap/recipe.jsonl`, 159 steps completed before the user stopped it): seed 0, batch 2, scale 2, AdamW peak 1e-4, weight decay 0.01, 300-step warmup from 1e-7, cosine to 1e-6 over 6144 steps, clip 1.0, real ANE block. 10-step means at steps 10–150: 5.137, 4.106, 4.502, 4.623, 3.952, 3.980, 3.522, 3.702, 3.594, 3.445, 3.512, 3.674, 3.413, 3.764, 3.191. Every step finite; step 61 loss 3.872, step 111 loss 3.232; the ladder never fired (318 of 318 row-backwards at peak 1); ANE forward vs GPU block cosine ≥ 0.99996. **Open issue**: during warmup the LoRA B factors are near zero, so the LoRA A gradients (which are proportional to B) are tiny and the fp16 per-factor cosine vs float32 was low early: worst-factor cosine 0.40–0.86 at steps 4–14, 0.95–0.98 at 15–25, with isolated dips to 0.71 (step 40) and 0.48 (step 46). d_hidden cosine stayed ≥ 0.9999 throughout. The acceptance criterion "per-factor cosine ≥ 0.99 on three real batches" therefore holds on GPU-state batches with the old recipe (section 5, paragraph 3) but has **not** been shown for the warmup phase of the new recipe. Possible remedies (proposed, untested): scale the A-factor outputs separately, or raise the cotangent peak for the LoRA-A gradient outputs only; or accept it, since Adam normalizes and the trajectory was healthy.

## 6. Recipe (applied in code this session; see section 9)

Values now in `configs/qlora_m4.yaml`: rank 8, `lora_scale: 2.0`, dropout 0.0, `num_layers: -1`, all 7 projections; AdamW `learning_rate: 0.0001`, `weight_decay: 0.01`, `max_grad_norm: 1.0`, `lr_schedule: {name: cosine_decay, warmup: 300, warmup_init: 1.0e-7, end: 1.0e-6}`. `finetune.resolve_schedule` turns that into the mlx-lm `build_schedule` config `{"name": "cosine_decay", "arguments": [1e-4, iters-300, 1e-6], "warmup": 300, "warmup_init": 1e-7}` with `iters` computed at run time (6144 at batch 2, 3 epochs). Clipping is `mlx.optimizers.clip_grad_norm(grad, 1.0)` in `train_loop.py::step` after `average_gradients` and the accumulation divide, before `optimizer.update`. Clipping cannot fix inf produced inside the ANE backward; that happens before grads reach the loop.

Why (**source-reported**): scale·lr was 20·1e-4 = 2e-3, 5–10x mlx-lm's own default pairing (scale 20 with lr 1e-5, mlx-lm discussion 1985, mlx-examples issue 1186), PEFT/Unsloth (alpha/r 1–2 at lr 2e-4, Unsloth lora-hyperparameters-guide), LLaMA-Factory Qwen3 (alpha = 2r, lr 1e-4, cosine, warmup_ratio 0.1, `examples/train_lora/qwen3_lora_sft.yaml`), ms-swift Qwen3 (r8/alpha32, lr 1e-4, warmup 0.05, `examples/train/think_model/qwen3_demo1.sh`), LoRA paper arXiv 2106.09685, QLoRA paper arXiv 2305.14314 (max_grad_norm 0.3), HF default clip 1.0. The GPU-only run was stable on the old recipe, so the recipe widens margin; the user's position is that it is not a substitute for the fp16 backward fix. My measurements (section 5) found the fp16 backward accurate at peak 1 and the divergence recipe-driven, so the two views should be reconciled by the next agent with the warmup-phase cosine issue in mind.

## 7. Resume procedure

a. Read `AGENTS.md`, `wiki/project/state.md`, the last entries of `wiki/log.md` (all 2026-10-28; none written by me), this file, `src/macreduce/ane_stack.py`, and the diff of the files in section 9.
b. `ps aux | rg main.py`; check `evidence/qwen3-0.6b-curve/latest`. Do not start a second `main.py`.
c. Decide on the warmup-phase LoRA-A cosine issue (section 5, last paragraph). If you change the ANE graph or the scaling, re-run `/tmp/anecheck.py` on `/tmp/cap/step-061.npz`, `step-110.npz`, `step-115.npz` (these have scale-20 factors baked in the npz but the engine bakes the model's scale, so re-capture with `/tmp/capture.py` after switching its `linear_to_lora_layers` call to scale 2.0) and the new-recipe replay `/tmp/replay.py 260 recipe`.
d. The recipe plumbing is done (section 9). Confirm with `.venv/bin/python -m pytest -q tests/test_config.py tests/test_checkpoint.py` and `MACREDUCE_TEST_MLX=1 .venv/bin/python -m pytest -q tests/test_config.py` (schedule values 1e-7 at step 0, 1e-4 at 300, ~1e-6 at 6143; passed tonight).
e. `.venv/bin/python -m pytest -q` (93 passed, 5 skipped tonight), `.venv/bin/ruff check .` (only the pre-existing E741 in immutable `raw/sources/S32.py` remains), `.venv/bin/python tools/wiki.py lint` (not run by me after my edits; I edited no wiki files).
f. Finish the ≥200-step new-recipe replay (`/tmp/replay.py 260 recipe`, about 2 s/step with the gradient check). Report finiteness, 10-step means, ladder count, and the cosine summary (`/tmp/cap/recipe.jsonl` has `grad_cos`, `grad_ratio`, `peaks`, `fwd_cos`, `lr` per step).
g. Append a `wiki/log.md` entry (never rewrite old ones) covering: rms_norm hypothesis tested and not confirmed; per-step ANE gradient accuracy; plumbing exactness; the fp32-block divergence experiments; the recipe change with the sources above; the warmup cosine issue. Update the QLoRA row in `wiki/project/state.md` and the one sentence in `wiki/guides/ane.md`. Keep measured / source-reported / inferred / proposed labels. `README.md` mentions only the command; the yaml comments were updated.
h. Launch exactly one detached trainer (section 2). First validation is ~1230 batches and takes several minutes; not a hang. Monitor the new directory's `run.log` and `history.jsonl` past step 300, ideally to the epoch checkpoint at step 2048 (35–60 min). On failure: `kill -TERM -<pid>`, confirm with `ps`, diagnose from that run's log, fix, relaunch. Do not delete run directories.

## 8. Constraints

Python 3.12 via `.venv/bin/python`; uv at `~/.local/bin/uv`. ruff line length 100, select E4 E7 E9 F I. Append-only wiki log. No silent GPU fallback (mask must be `0x4`). Zero is not missing telemetry; `ane_active` can stay 0 while the engine runs; mactop is system-wide. Perplexity is the Qwen loss (GPU base plus one ANE block), not an ANE-only result. Do not flatten the `macreduce` package. Do not edit immutable `raw/sources` or historical evidence directories. Figures refresh every 60 s inside the run (`curve.run_curve`); `macreduce figures <dir>` redraws. Memory limit is the MLX recommended working set, auto = 19069665280 bytes. `np.array` on a bfloat16 mlx array fails; cast with `.astype(mx.float32)` first. Resume checks (`train_loop._check_resume`) now reject checkpoints whose recipe differs, so old-recipe checkpoints (there are none) cannot be resumed under the new yaml.

## 9. What I changed in this session

Code and config (all uncommitted in the working tree; `git status` also shows many pre-existing uncommitted changes from earlier sessions, so do not assume everything modified is mine):

- `src/macreduce/finetune.py`: added `DEFAULT_RECIPE`, `recipe_from_config(config)`, `validate_recipe(recipe)`, `resolve_schedule(schedule, learning_rate, iters)`; `run_lora`/`_run_lora` take `recipe`; `_namespace(..., recipe=None)` fills `lora_parameters`, `lr_schedule`, `max_grad_norm`, `weight_decay`, `optimizer_config`; the report dict now records `lora` from the recipe plus `weight_decay`, `max_grad_norm`, `lr_schedule`.
- `src/macreduce/train_loop.py`: AdamW built from `build_schedule(resolve_schedule(...))` when a schedule is set, `weight_decay` passed, `clip_grad_norm` in `step`, recipe fields in the resume `identity`, `_check_resume` compares them.
- `src/macreduce/batch_fit.py`: `choose_batch(..., recipe=None)` threads the recipe through `_sweep` → `_child` spec → `run_probe` → `_namespace`; probe AdamW gets the weight decay.
- `main.py`: passes `recipe_from_config(config)` to `choose_batch` and `run_curve` (which forwards kwargs to `run_lora`).
- `configs/qlora_m4.yaml`: scale 2.0, weight_decay, max_grad_norm, lr_schedule, updated comments.
- `tests/test_config.py`: asserts the new yaml values; `test_yaml_recipe_reaches_the_training_namespace`; `test_recipe_rejects_bad_values`; `test_schedule_hits_warmup_peak_and_end` (opt-in via `MACREDUCE_TEST_MLX=1`).
- `tests/test_checkpoint.py`: `test_resume_identity_checks_the_recipe`.

Not changed: `src/macreduce/ane_stack.py` (no graph or ladder change), `tests/test_ane_stack.py`, any wiki page, any evidence directory, `README.md`.

`/tmp` scripts worth keeping (they will vanish on reboot; copy them under `/tmp` again from this description if lost):
- `/tmp/capture.py` — GPU-only replay that captures block-27 hidden/cotangent/factors and float32 reference grads (`fp32_block` helper is imported by the others). Still uses scale 20.0; change the `linear_to_lora_layers` call for the new recipe.
- `/tmp/anecheck.py` — ANE backward vs float32 per factor at several cotangent peaks on the captured npz files.
- `/tmp/rowcheck.py` — per-position d_hidden accuracy.
- `/tmp/intermediates.py` — float64 forward/backward intermediate magnitudes of block 27 for a capture or failure dump.
- `/tmp/replay.py N tag` — instrumented GPU+ANE replay under the NEW recipe (schedule, clip, weight decay), logs `/tmp/cap/<tag>.jsonl`, dumps `/tmp/cap/fail-<tag>-<step>.npz` on failure.
- `/tmp/fake_replay.py N {fp32|plumb|real}` and `/tmp/plumb.py` — plumbing and precision experiments (old recipe).
- `/tmp/chaos.py N {noise[:layer:eps]|fp32block[:layer]}` — GPU-only perturbation runs (old recipe).
- `/tmp/rms_probe.py` — isolated rms_norm VJP overflow probe.
- Data: `/tmp/cap/step-*.npz`, `/tmp/cap/fail-base-111.npz`, `/tmp/cap/*.jsonl`, `/tmp/cap/*_losses.npy`.
