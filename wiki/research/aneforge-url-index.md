---
title: "ANEForge site and paper index"
type: research-ledger
status: active
updated: 2026-10-29
sources: ["S10", "S38", "S39", "S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47", "S48", "S49", "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57", "S58", "S59", "S60", "S61", "S62", "S63", "S64", "S65", "S66", "S67", "S68", "S69", "S70", "S71", "S72", "S73", "S74", "S75", "S76", "S77", "S78", "S79", "S80", "S81", "S82", "S83", "S84", "S85", "S86", "S87", "S88", "S89", "S90", "S91", "S92", "S93", "S94", "S95", "S96", "S97", "S98", "S99", "S100", "S101", "S102", "S103", "S104", "S105", "S106", "S107", "S108", "S109", "S110", "S111", "S112", "S113", "S114", "S115", "S116", "S117", "S118", "S119", "S120", "S121", "S122", "S123", "S124", "S125", "S158", "S159", "S160", "S161"]
---

# ANEForge site and paper index

This page is the retrieval index for the ANEForge sites and for arXiv:2606.17090. Search [ANEForge lookup](../queries/aneforge.md) before writing a call. The procedures are in [Write an ANEForge program](../guides/aneforge.md). The paper abstract was read on [S39](../sources/S39.md). The PDF body in [S38](../sources/S38.md) was not text-extracted. [S10](../sources/S10.md) remains the earlier raw copy of `docs/training.md` and was not replaced. The published training page is [S76](../sources/S76.md).

The file listing is pinned to the tree the GitHub API returned for `main` on this retrieval. That response has sha `896c4943c5ebf811c0e815f8d1381fd1e157fa7d` and `truncated: false`. A separate request for the `main` commit returned the same sha, with committer date 2026-10-07. `main` can move. The sha on [S41](../sources/S41.md) is the listing that was saved.

## What was asked for, and what the sites actually publish

`https://aneforge.com/sitemap.xml`, `https://aneforge.com/sitemap_index.xml`, and `https://aneforge.com/robots.txt` each returned HTTP 404. The host is one HTML document. A link walk from that page found no second HTML page on the host. Fonts and favicons were not saved.

`https://aneforge.readthedocs.io/sitemap.xml` and `https://aneforge.readthedocs.io/en/latest/sitemap.xml` each returned an empty urlset (109 bytes). The 35 documentation pages below were found by walking links from `https://aneforge.readthedocs.io/en/latest/`.

`https://ane-guide.readthedocs.io/sitemap.xml` lists only `https://ane-guide.readthedocs.io/en/latest/`. `https://ane-guide.readthedocs.io/en/latest/sitemap.xml` returned HTTP 404. The 48 chapter pages were taken from `doc_urls` in `https://ane-guide.readthedocs.io/en/latest/searchindex.js`, then fetched directly. All 49 guide URLs, including the landing page, returned HTTP 200.

GitHub directory pages were walked from `https://github.com/sbryngelson/ANEForge` and `https://github.com/sbryngelson/ANEForge/tree/main/examples`, staying on `/tree/main` paths. Markdown blob pages were also reached. They were not copied again: the rendered docs are the Read the Docs snapshots, and every path is in the file tree. Example programs are listed from that tree and were not copied as their own snapshots.

## Paper

- [S38](../sources/S38.md) — <https://arxiv.org/pdf/2606.17090>
- [S39](../sources/S39.md) — <https://arxiv.org/abs/2606.17090>

Reported by source. Spencer H. Bryngelson. Submitted 12 Jun 2026, revised 15 Oct 2026 (v2). Primary category cs.PL, with cs.AI and cs.MS on the abs page. The abstract says ANEForge compiles a lazy tensor graph of 58 fused operators and 19 native bridge operators into one Neural Engine program, dispatched through the same daemon and kernel stack as Apple's internal framework, without Core ML. It says the package can run a forward, a backward, and an optimizer update on the engine, keep decoder and optimizer state resident, and stream int8, int4, and sparse weights. It reports a small fused call near 90 microseconds against a stated 70 microsecond dispatch floor, a ResNet-18 forward at 0.33 ms, and end-to-end checks for ResNet-18, a sentence encoder, a Vision Transformer, and a Stable Diffusion U-Net forward. It says the target is Apple Silicon on macOS 14 and later, checked per release against a recorded macOS and compiler version. The experimental HTML at `https://arxiv.org/html/2606.17090v2` was not saved. The DOI on the abs page is `https://doi.org/10.48550/arXiv.2606.17090`.

## aneforge.com

- [S40](../sources/S40.md) — <https://aneforge.com/>

The page title is “ANEForge: program the Apple Neural Engine directly, from Python”. The links that leave the page are the two documentation sites, the GitHub repository, the paper, Zenodo `https://doi.org/10.5281/zenodo.20672609`, the Hugging Face space `https://huggingface.co/spaces/aneforge/ane-leaderboard`, the examples `fluid_vorticity.py`, `reaction_diffusion.py`, and `train_neural_ca.py`, and the `bench/mlperf` tree. Those three programs and the leaderboard and Zenodo record were not snapshotted. The example paths are in the file list below. Numbers printed on the homepage were not checked.

## aneforge.readthedocs.io

- [S42](../sources/S42.md) — <https://aneforge.readthedocs.io/en/latest>
- [S43](../sources/S43.md) — <https://aneforge.readthedocs.io/en/latest/aneforge-api>
- [S44](../sources/S44.md) — <https://aneforge.readthedocs.io/en/latest/api>
- [S45](../sources/S45.md) — <https://aneforge.readthedocs.io/en/latest/api/compile>
- [S46](../sources/S46.md) — <https://aneforge.readthedocs.io/en/latest/api/graph>
- [S47](../sources/S47.md) — <https://aneforge.readthedocs.io/en/latest/api/math>
- [S48](../sources/S48.md) — <https://aneforge.readthedocs.io/en/latest/api/models>
- [S49](../sources/S49.md) — <https://aneforge.readthedocs.io/en/latest/api/training>
- [S50](../sources/S50.md) — <https://aneforge.readthedocs.io/en/latest/capabilities>
- [S51](../sources/S51.md) — <https://aneforge.readthedocs.io/en/latest/cross-chip>
- [S52](../sources/S52.md) — <https://aneforge.readthedocs.io/en/latest/developer/autograd>
- [S53](../sources/S53.md) — <https://aneforge.readthedocs.io/en/latest/developer/bridges>
- [S54](../sources/S54.md) — <https://aneforge.readthedocs.io/en/latest/developer/capabilities-and-targets>
- [S55](../sources/S55.md) — <https://aneforge.readthedocs.io/en/latest/developer/compile-pipeline>
- [S56](../sources/S56.md) — <https://aneforge.readthedocs.io/en/latest/developer/models>
- [S57](../sources/S57.md) — <https://aneforge.readthedocs.io/en/latest/developer/numerics>
- [S58](../sources/S58.md) — <https://aneforge.readthedocs.io/en/latest/developer/op-quirks>
- [S59](../sources/S59.md) — <https://aneforge.readthedocs.io/en/latest/developer/optimizer-and-cost>
- [S60](../sources/S60.md) — <https://aneforge.readthedocs.io/en/latest/developer/overview>
- [S61](../sources/S61.md) — <https://aneforge.readthedocs.io/en/latest/development>
- [S62](../sources/S62.md) — <https://aneforge.readthedocs.io/en/latest/dispatch>
- [S63](../sources/S63.md) — <https://aneforge.readthedocs.io/en/latest/e5rt-dispatch-reference>
- [S64](../sources/S64.md) — <https://aneforge.readthedocs.io/en/latest/faq>
- [S65](../sources/S65.md) — <https://aneforge.readthedocs.io/en/latest/getting-started>
- [S66](../sources/S66.md) — <https://aneforge.readthedocs.io/en/latest/glossary>
- [S67](../sources/S67.md) — <https://aneforge.readthedocs.io/en/latest/llm>
- [S68](../sources/S68.md) — <https://aneforge.readthedocs.io/en/latest/mil-primer>
- [S69](../sources/S69.md) — <https://aneforge.readthedocs.io/en/latest/mlperf>
- [S70](../sources/S70.md) — <https://aneforge.readthedocs.io/en/latest/numeric-cliffs>
- [S71](../sources/S71.md) — <https://aneforge.readthedocs.io/en/latest/onnx>
- [S72](../sources/S72.md) — <https://aneforge.readthedocs.io/en/latest/op-catalog>
- [S73](../sources/S73.md) — <https://aneforge.readthedocs.io/en/latest/reproducibility>
- [S74](../sources/S74.md) — <https://aneforge.readthedocs.io/en/latest/roadmap>
- [S75](../sources/S75.md) — <https://aneforge.readthedocs.io/en/latest/rooflines>
- [S76](../sources/S76.md) — <https://aneforge.readthedocs.io/en/latest/training>

[S76](../sources/S76.md) is the rendered training page. [S10](../sources/S10.md) is an older raw copy of `docs/training.md` from `raw.githubusercontent.com`, retrieved 2026-10-23. S76 is a new HTML snapshot of the published page.

## ane-guide.readthedocs.io

- [S77](../sources/S77.md) — <https://ane-guide.readthedocs.io/en/latest/>
- [S78](../sources/S78.md) — <https://ane-guide.readthedocs.io/en/latest/appendices/a-op-device-matrix.html>
- [S79](../sources/S79.md) — <https://ane-guide.readthedocs.io/en/latest/appendices/b-hidden-layer-catalog.html>
- [S80](../sources/S80.md) — <https://ane-guide.readthedocs.io/en/latest/appendices/c-decoded-tables.html>
- [S81](../sources/S81.md) — <https://ane-guide.readthedocs.io/en/latest/appendices/d-glossary.html>
- [S82](../sources/S82.md) — <https://ane-guide.readthedocs.io/en/latest/appendices/f-provenance.html>
- [S83](../sources/S83.md) — <https://ane-guide.readthedocs.io/en/latest/back-matter/methodology.html>
- [S84](../sources/S84.md) — <https://ane-guide.readthedocs.io/en/latest/back-matter/open-questions.html>
- [S85](../sources/S85.md) — <https://ane-guide.readthedocs.io/en/latest/back-matter/revision-history.html>
- [S86](../sources/S86.md) — <https://ane-guide.readthedocs.io/en/latest/back-matter/statements.html>
- [S87](../sources/S87.md) — <https://ane-guide.readthedocs.io/en/latest/intro.html>
- [S88](../sources/S88.md) — <https://ane-guide.readthedocs.io/en/latest/part-1-machine/01-what-the-ane-is.html>
- [S89](../sources/S89.md) — <https://ane-guide.readthedocs.io/en/latest/part-1-machine/02-execution-model.html>
- [S90](../sources/S90.md) — <https://ane-guide.readthedocs.io/en/latest/part-1-machine/03-numerics.html>
- [S91](../sources/S91.md) — <https://ane-guide.readthedocs.io/en/latest/part-1-machine/04-capability-surface.html>
- [S92](../sources/S92.md) — <https://ane-guide.readthedocs.io/en/latest/part-2-reaching/05-software-stack.html>
- [S93](../sources/S93.md) — <https://ane-guide.readthedocs.io/en/latest/part-2-reaching/06-dispatch-without-coreml.html>
- [S94](../sources/S94.md) — <https://ane-guide.readthedocs.io/en/latest/part-2-reaching/07-weights-and-compression.html>
- [S95](../sources/S95.md) — <https://ane-guide.readthedocs.io/en/latest/part-2-reaching/08-entitlement-boundary.html>
- [S96](../sources/S96.md) — <https://ane-guide.readthedocs.io/en/latest/part-3-performance/09-roofline.html>
- [S97](../sources/S97.md) — <https://ane-guide.readthedocs.io/en/latest/part-3-performance/10-power-efficiency.html>
- [S98](../sources/S98.md) — <https://ane-guide.readthedocs.io/en/latest/part-3-performance/11-ane-vs-gpu-vs-cpu.html>
- [S99](../sources/S99.md) — <https://ane-guide.readthedocs.io/en/latest/part-3-performance/12-cross-chip.html>
- [S100](../sources/S100.md) — <https://ane-guide.readthedocs.io/en/latest/part-4-workloads/13-vision-conv-encoders.html>
- [S101](../sources/S101.md) — <https://ane-guide.readthedocs.io/en/latest/part-4-workloads/14-llm-case-study.html>
- [S102](../sources/S102.md) — <https://ane-guide.readthedocs.io/en/latest/part-4-workloads/15-training.html>
- [S103](../sources/S103.md) — <https://ane-guide.readthedocs.io/en/latest/part-4-workloads/16-numerical-scientific.html>
- [S104](../sources/S104.md) — <https://ane-guide.readthedocs.io/en/latest/part-5-practice/17-model-design-rules.html>
- [S105](../sources/S105.md) — <https://ane-guide.readthedocs.io/en/latest/part-5-practice/18-optimization-cost-model.html>
- [S106](../sources/S106.md) — <https://ane-guide.readthedocs.io/en/latest/part-5-practice/19-pitfalls-and-limits.html>
- [S107](../sources/S107.md) — <https://ane-guide.readthedocs.io/en/latest/part-5-practice/handoff.html>
- [S108](../sources/S108.md) — <https://ane-guide.readthedocs.io/en/latest/part-6-silicon/20-datapath-mac.html>
- [S109](../sources/S109.md) — <https://ane-guide.readthedocs.io/en/latest/part-6-silicon/21-memory-hierarchy.html>
- [S110](../sources/S110.md) — <https://ane-guide.readthedocs.io/en/latest/part-7-toolchain/22-compiler.html>
- [S111](../sources/S111.md) — <https://ane-guide.readthedocs.io/en/latest/part-7-toolchain/23-program-format.html>
- [S112](../sources/S112.md) — <https://ane-guide.readthedocs.io/en/latest/part-7-toolchain/24-hal-gates.html>
- [S113](../sources/S113.md) — <https://ane-guide.readthedocs.io/en/latest/part-7-toolchain/25-compression-internals.html>
- [S114](../sources/S114.md) — <https://ane-guide.readthedocs.io/en/latest/part-7-toolchain/26-hidden-layers.html>
- [S115](../sources/S115.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/27-kernel-driver.html>
- [S116](../sources/S116.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/28-dart-iommu.html>
- [S117](../sources/S117.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/29-firmware.html>
- [S118](../sources/S118.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/30-csne-protocol.html>
- [S119](../sources/S119.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/31-power-thermal.html>
- [S120](../sources/S120.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/32-security.html>
- [S121](../sources/S121.md) — <https://ane-guide.readthedocs.io/en/latest/part-8-system-internals/33-telemetry.html>
- [S122](../sources/S122.md) — <https://ane-guide.readthedocs.io/en/latest/part-9-cross-silicon/34-the-28-targets.html>
- [S123](../sources/S123.md) — <https://ane-guide.readthedocs.io/en/latest/part-9-cross-silicon/35-per-family-codegen.html>
- [S124](../sources/S124.md) — <https://ane-guide.readthedocs.io/en/latest/part-9-cross-silicon/36-upper-tier.html>
- [S125](../sources/S125.md) — <https://ane-guide.readthedocs.io/en/latest/references.html>

## GitHub

The directory HTML for fonts, badges, workflows, and folder listings was removed. Those pages repeated [S41](../sources/S41.md) and were not an API. Four files from commit `896c4943c5ebf811c0e815f8d1381fd1e157fa7d` were saved and read:

- [S158](../sources/S158.md) — `examples/README.md`
- [S159](../sources/S159.md) — `examples/demos/training_on_ane.py`
- [S160](../sources/S160.md) — `examples/demos/resident_state.py`
- [S161](../sources/S161.md) — `examples/llama_block_causal.py`

## File tree at the saved sha

[S41](../sources/S41.md) lists 473 git entries, of which 443 are files. Sizes are the sizes in that tree response. The file bodies were not read except [S158](../sources/S158.md), [S159](../sources/S159.md), [S160](../sources/S160.md), and [S161](../sources/S161.md), and the rendered docs above.

### examples/

| Path | Bytes |
|---|---:|
| `examples/README.md` | 8574 |
| `examples/_common.py` | 4838 |
| `examples/_rag.py` | 2306 |
| `examples/autotune.py` | 9879 |
| `examples/benchmarks/README.md` | 488 |
| `examples/benchmarks/bench_encoder_batched.py` | 4294 |
| `examples/benchmarks/bench_encoder_gpu.py` | 2803 |
| `examples/benchmarks/rank_worker_bench.py` | 3078 |
| `examples/benchmarks/sdpa_worker_bench.py` | 2764 |
| `examples/benchmarks/topk_worker_bench.py` | 2685 |
| `examples/cifar_data.py` | 1630 |
| `examples/clip_zero_shot.py` | 4267 |
| `examples/compress_weights.py` | 4751 |
| `examples/data/mnist_subset.npz` | 331206 |
| `examples/demo.py` | 5537 |
| `examples/demos/README.md` | 3893 |
| `examples/demos/ane_vs_gpu_cpu.py` | 1881 |
| `examples/demos/batching_amortization.py` | 1209 |
| `examples/demos/capability_surface.py` | 1279 |
| `examples/demos/chaining_depth.py` | 1472 |
| `examples/demos/cross_chip_cost_model.py` | 1547 |
| `examples/demos/dispatch_no_coreml.py` | 1425 |
| `examples/demos/entitlement_boundary.py` | 1441 |
| `examples/demos/execution_model_floor.py` | 1363 |
| `examples/demos/hidden_layers.py` | 1786 |
| `examples/demos/llm_attention_kvcache.py` | 2070 |
| `examples/demos/mil_dialect.py` | 1092 |
| `examples/demos/numerical_scientific.py` | 1555 |
| `examples/demos/numerics_fp16.py` | 1720 |
| `examples/demos/optimization_autotune.py` | 1690 |
| `examples/demos/pitfalls_limits.py` | 2159 |
| `examples/demos/power_efficiency.py` | 1519 |
| `examples/demos/resident_state.py` | 1262 |
| `examples/demos/roofline_bandwidth.py` | 1375 |
| `examples/demos/roofline_compute.py` | 1387 |
| `examples/demos/single_in_flight.py` | 1728 |
| `examples/demos/training_on_ane.py` | 1786 |
| `examples/demos/vision_conv_encoder.py` | 2171 |
| `examples/demos/weights_compression.py` | 1762 |
| `examples/demos/what_the_ane_is.py` | 1352 |
| `examples/demos/zero_copy_io.py` | 1714 |
| `examples/eigenvalues_svd.py` | 2862 |
| `examples/factorize.py` | 1471 |
| `examples/fft.py` | 1304 |
| `examples/fluid_vorticity.py` | 11230 |
| `examples/game_of_life.py` | 5944 |
| `examples/gpt2.py` | 5447 |
| `examples/gpt_generate_ane.py` | 8394 |
| `examples/gpt_multilayer_resident.py` | 5805 |
| `examples/heat_equation.py` | 4253 |
| `examples/llama_block_causal.py` | 3180 |
| `examples/llm_chat.py` | 2363 |
| `examples/llm_prefill.py` | 4941 |
| `examples/lorenz.py` | 5858 |
| `examples/make_mnist_subset.py` | 2191 |
| `examples/mandelbrot.py` | 6613 |
| `examples/moe_chat.py` | 3421 |
| `examples/native_geometry.py` | 3059 |
| `examples/native_norms.py` | 2191 |
| `examples/native_pixel_ops.py` | 4476 |
| `examples/native_ranking.py` | 1634 |
| `examples/nbody.py` | 3797 |
| `examples/onnx_finetune.py` | 3212 |
| `examples/onnx_import.py` | 2061 |
| `examples/paired_fp16.py` | 9352 |
| `examples/pointcloud.py` | 3048 |
| `examples/poisson_spectral.py` | 3180 |
| `examples/quickstart.py` | 3378 |
| `examples/qwen35_chat.py` | 4280 |
| `examples/rag_chat.py` | 9064 |
| `examples/rag_embeddings.py` | 1722 |
| `examples/reaction_diffusion.py` | 8942 |
| `examples/resnet18.py` | 1022 |
| `examples/sd15.py` | 21873 |
| `examples/sd_unet.py` | 5546 |
| `examples/sd_vae.py` | 3707 |
| `examples/sdpa.py` | 2077 |
| `examples/sentence_embeddings.py` | 2344 |
| `examples/sentence_transformers_ane.py` | 1136 |
| `examples/solve_linear_systems.py` | 1250 |
| `examples/spec_chat.py` | 3043 |
| `examples/spectral_analysis.py` | 4507 |
| `examples/superres_espcn.py` | 2861 |
| `examples/train_charlm.py` | 4126 |
| `examples/train_charlm_corpus.py` | 4413 |
| `examples/train_charlm_deep.py` | 6120 |
| `examples/train_cifar_cnn.py` | 4180 |
| `examples/train_llama_block.py` | 2382 |
| `examples/train_mnist_cnn.py` | 2425 |
| `examples/train_mnist_mlp.py` | 2227 |
| `examples/train_neural_ca.py` | 10916 |
| `examples/train_transformer.py` | 4062 |
| `examples/train_transformer_prenorm.py` | 2627 |
| `examples/vit.py` | 9156 |
| `examples/whisper.py` | 4090 |

### .githooks

| Path | Bytes |
|---|---:|
| `.githooks/pre-commit` | 1395 |

### .github

| Path | Bytes |
|---|---:|
| `.github/badges/loc.json` | 79 |
| `.github/dependabot.yml` | 530 |
| `.github/workflows/ci.yml` | 7327 |
| `.github/workflows/loc.yml` | 1463 |
| `.github/workflows/pages.yml` | 1171 |
| `.github/workflows/release.yml` | 2158 |
| `.github/workflows/sync-hf-dataset.yml` | 726 |

### .gitignore

| Path | Bytes |
|---|---:|
| `.gitignore` | 3212 |

### .pre-commit-config.yaml

| Path | Bytes |
|---|---:|
| `.pre-commit-config.yaml` | 1513 |

### .pylintrc

| Path | Bytes |
|---|---:|
| `.pylintrc` | 675 |

### .readthedocs.yaml

| Path | Bytes |
|---|---:|
| `.readthedocs.yaml` | 301 |

### AGENTS.md

| Path | Bytes |
|---|---:|
| `AGENTS.md` | 5858 |

### CHANGELOG.md

| Path | Bytes |
|---|---:|
| `CHANGELOG.md` | 286 |

### CITATION.cff

| Path | Bytes |
|---|---:|
| `CITATION.cff` | 804 |

### CONTRIBUTING.md

| Path | Bytes |
|---|---:|
| `CONTRIBUTING.md` | 2481 |

### LICENSE

| Path | Bytes |
|---|---:|
| `LICENSE` | 1078 |

### NOTICE

| Path | Bytes |
|---|---:|
| `NOTICE` | 670 |

### README.md

| Path | Bytes |
|---|---:|
| `README.md` | 18170 |

### SECURITY.md

| Path | Bytes |
|---|---:|
| `SECURITY.md` | 1998 |

### aneforge

| Path | Bytes |
|---|---:|
| `aneforge/__init__.py` | 5094 |
| `aneforge/_blob.py` | 3984 |
| `aneforge/_bridges/__init__.py` | 154 |
| `aneforge/_bridges/_netplist.py` | 95719 |
| `aneforge/_bridges/ane_cost_volume_fused.py` | 2515 |
| `aneforge/_bridges/ane_cross_correlation_fused.py` | 2304 |
| `aneforge/_bridges/ane_cross_product_fused.py` | 1773 |
| `aneforge/_bridges/ane_dynamic_slice_fused.py` | 1881 |
| `aneforge/_bridges/ane_fps_fused.py` | 2562 |
| `aneforge/_bridges/ane_input_view_fused.py` | 2069 |
| `aneforge/_bridges/ane_radius_search_fused.py` | 2576 |
| `aneforge/_bridges/ane_rank_fused.py` | 6651 |
| `aneforge/_bridges/ane_rearrange_fused.py` | 8663 |
| `aneforge/_bridges/ane_sdpa_fused.py` | 6463 |
| `aneforge/_bridges/ane_structural_fused.py` | 3661 |
| `aneforge/_bridges/lrn_fused.py` | 3228 |
| `aneforge/_bridges/minmax_norm_fused.py` | 3171 |
| `aneforge/_bridges/scaled_elementwise_fused.py` | 3065 |
| `aneforge/_capabilities.py` | 43294 |
| `aneforge/_circuit.py` | 2120 |
| `aneforge/_compile.py` | 82155 |
| `aneforge/_cost.py` | 26449 |
| `aneforge/_gguf.py` | 2060 |
| `aneforge/_invokers/README.md` | 1088 |
| `aneforge/_invokers/layer_invoker.mm` | 22209 |
| `aneforge/_invokers/persistent_worker.mm` | 22191 |
| `aneforge/_invokers/rank_invoker.mm` | 16961 |
| `aneforge/_invokers/sdpa_invoker.mm` | 20448 |
| `aneforge/_lib/ane_e5rt_dispatch.mm` | 45913 |
| `aneforge/_lib/build.sh` | 580 |
| `aneforge/_lib/e5rt_api.h` | 7555 |
| `aneforge/_netplist_worker.py` | 10105 |
| `aneforge/_op_catalog.py` | 48568 |
| `aneforge/_optimize.py` | 30826 |
| `aneforge/_paired.py` | 5122 |
| `aneforge/_rewrite.py` | 12478 |
| `aneforge/_runtime.py` | 13038 |
| `aneforge/_targets.py` | 13825 |
| `aneforge/ane_cost_model.json` | 47049 |
| `aneforge/autograd.py` | 45654 |
| `aneforge/build.py` | 2670 |
| `aneforge/costmodel_curves.json` | 14357 |
| `aneforge/dsp.py` | 32183 |
| `aneforge/einsum.py` | 16692 |
| `aneforge/fft.py` | 17490 |
| `aneforge/full_mil_vocabulary_sweep.json` | 39366 |
| `aneforge/graph.py` | 55314 |
| `aneforge/linalg.py` | 49564 |
| `aneforge/llm.py` | 48725 |
| `aneforge/models.py` | 57689 |
| `aneforge/moe.py` | 11308 |
| `aneforge/onnx.py` | 67668 |
| `aneforge/py.typed` | 0 |
| `aneforge/qwen35.py` | 12125 |
| `aneforge/sentence_transformers.py` | 2700 |
| `aneforge/special.py` | 19161 |
| `aneforge/speculative.py` | 10428 |
| `aneforge/streaming.py` | 3908 |

### bench

| Path | Bytes |
|---|---:|
| `bench/README.md` | 3125 |
| `bench/_ane_rail.py` | 12803 |
| `bench/_machine.py` | 18036 |
| `bench/_mil.py` | 843 |
| `bench/_probe.py` | 3364 |
| `bench/aggregate_rooflines.py` | 20044 |
| `bench/below_ridge_fusion.py` | 7970 |
| `bench/compress_speedup_bench.py` | 3634 |
| `bench/cross_path_compress_bench.py` | 3772 |
| `bench/decode_int8_accuracy.py` | 3369 |
| `bench/decode_measurement.py` | 20192 |
| `bench/device_bandwidth_roofline.py` | 16862 |
| `bench/device_compare.py` | 3896 |
| `bench/device_compare_wattcomplete.py` | 37005 |
| `bench/device_saturation_sweep.py` | 17728 |
| `bench/device_serving_sweep.py` | 23402 |
| `bench/encoder_serving_crosspath.py` | 9046 |
| `bench/fused_gpu_baseline.py` | 17943 |
| `bench/gemv_bandwidth_sweep.py` | 9790 |
| `bench/layer_norm_compile_probe.py` | 7940 |
| `bench/mlperf/.gitignore` | 108 |
| `bench/mlperf/README.md` | 14816 |
| `bench/mlperf/compare/compare.py` | 4487 |
| `bench/mlperf/compare/competitors.csv` | 668 |
| `bench/mlperf/llm_decode.py` | 2495 |
| `bench/mlperf/loadgen_lite.py` | 8617 |
| `bench/mlperf/loadgen_official.py` | 6504 |
| `bench/mlperf/repro.py` | 5024 |
| `bench/mlperf/repro.sh` | 1982 |
| `bench/mlperf/resnet50.py` | 12451 |
| `bench/mlperf/results/comparison.md` | 2112 |
| `bench/mlperf/results/llm_decode_fp16.json` | 531 |
| `bench/mlperf/results/llm_decode_int8.json` | 534 |
| `bench/mlperf/results/loadgen/.gitignore` | 142 |
| `bench/mlperf/results/loadgen/mlperf_log_summary.txt` | 1821 |
| `bench/mlperf/results/resnet50_accuracy_closed_val50k.json` | 455 |
| `bench/mlperf/results/resnet50_accuracy_normalized.json` | 501 |
| `bench/mlperf/results/resnet50_accuracy_sample1000.json` | 506 |
| `bench/mlperf/results/resnet50_accuracy_val50k.json` | 450 |
| `bench/mlperf/results/resnet50_fidelity.json` | 323 |
| `bench/mlperf/results/resnet50_loadgen_singlestream.json` | 878 |
| `bench/mlperf/results/resnet50_official.json` | 484 |
| `bench/mlperf/results/submission_checker_v5.1.txt` | 30340 |
| `bench/mlperf/run.py` | 5842 |
| `bench/mlperf/run_accuracy.py` | 6037 |
| `bench/mlperf/run_llm.py` | 2558 |
| `bench/mlperf/run_official.py` | 4149 |
| `bench/mlperf/run_submission.py` | 13420 |
| `bench/model_int4_bench.py` | 9357 |
| `bench/numeric_cliffs.py` | 6297 |
| `bench/opt1_int8_saturation_repro.py` | 2809 |
| `bench/real_models_fp16.py` | 8717 |
| `bench/results/ROOFLINES.md` | 6005 |
| `bench/results/below_ridge_fusion.json` | 2202 |
| `bench/results/compress_speedup_bench.json` | 13996 |
| `bench/results/cross_path_compress_bench.json` | 4419 |
| `bench/results/decode_int8_accuracy.json` | 720 |
| `bench/results/decode_measurement_results.json` | 15376 |
| `bench/results/device_bandwidth_roofline_results.json` | 44391 |
| `bench/results/device_compare_wattcomplete_results.json` | 34629 |
| `bench/results/device_compare_wattcomplete_results_M1.json` | 29712 |
| `bench/results/device_compare_wattcomplete_results_M2.json` | 34583 |
| `bench/results/device_compare_wattcomplete_results_M5.json` | 34629 |
| `bench/results/device_saturation_sweep_results.json` | 21240 |
| `bench/results/device_serving_sweep_results.json` | 63763 |
| `bench/results/encoder_serving_crosspath.json` | 6964 |
| `bench/results/fused_gpu_baseline_results.json` | 10100 |
| `bench/results/gemv_bandwidth_sweep_results.json` | 10513 |
| `bench/results/model_int4_bench.json` | 4341 |
| `bench/results/numeric_cliffs_results.json` | 1048 |
| `bench/results/real_models_fp16_results.json` | 2613 |
| `bench/results/roofline_analysis_results.json` | 26302 |
| `bench/results/rooflines.json` | 7481 |
| `bench/results/rooflines/README.md` | 3748 |
| `bench/results/rooflines/roofline-apple-m1-MacBookAir10_1-bb9e71be3ae6-22b6e4fa.json` | 116358 |
| `bench/results/rooflines/roofline-apple-m1-MacBookPro17_1-dcf5b8a7dd09-4b3e72c0.json` | 92735 |
| `bench/results/rooflines/roofline-apple-m1-MacBookPro17_1-dcf5b8a7dd09-b1ab0125.json` | 83849 |
| `bench/results/rooflines/roofline-apple-m1-max-MacBookPro18_2-8dc9588b5db6-2d0173cb.json` | 89316 |
| `bench/results/rooflines/roofline-apple-m1-max-MacBookPro18_2-8dc9588b5db6-85ebfafa.json` | 91542 |
| `bench/results/rooflines/roofline-apple-m2-max-Mac14_6-8c6229361be7-773f5b28.json` | 94283 |
| `bench/results/rooflines/roofline-apple-m2-pro-Mac14_12-723396ad9091-370b3d53.json` | 83407 |
| `bench/results/rooflines/roofline-apple-m2-pro-Mac14_12-723396ad9091-44d76599.json` | 115364 |
| `bench/results/rooflines/roofline-apple-m4-Mac16_12-d4109cd41a75-6779bad2.json` | 234454 |
| `bench/results/rooflines/roofline-apple-m4-max-Mac16_6-c77b1414be16-d11ec8da.json` | 234655 |
| `bench/results/rooflines/roofline-apple-m4-pro-Mac16_7-7fdf43ae2309-9094e75f.json` | 116607 |
| `bench/results/rooflines/roofline-apple-m5-Mac17_3-019d3ac27339-426ca052.json` | 93053 |
| `bench/results/rooflines/roofline-apple-m5-Mac17_3-019d3ac27339-dd5e5c95.json` | 92766 |
| `bench/results/rooflines/roofline-apple-m5-pro-Mac17_8-c38210cfc8ea-a154b89e.json` | 199379 |
| `bench/results/rooflines/roofline-apple-m6-Mac18_5-8701b924e26c-3fab2f9b.json` | 88698 |
| `bench/rms_norm_compile_probe.py` | 2563 |
| `bench/roofline_analysis.py` | 24868 |
| `bench/roofline_suite.py` | 11666 |
| `bench/whisper_encoder_ane/README.md` | 9811 |
| `bench/whisper_encoder_ane/bench_energy.py` | 3724 |
| `bench/whisper_encoder_ane/bench_latency.py` | 2415 |
| `bench/whisper_encoder_ane/encoder.py` | 8931 |
| `bench/whisper_encoder_ane/export_bundle.py` | 5348 |
| `bench/whisper_encoder_ane/fidelity.py` | 990 |
| `bench/whisper_encoder_ane/run_cpp.sh` | 1158 |
| `bench/whisper_encoder_ane/validate_real.py` | 2188 |
| `bench/whisper_encoder_ane/wer_proxy.py` | 3007 |
| `bench/whisper_encoder_ane/whisper_ane_run.cpp` | 4960 |

### docs

| Path | Bytes |
|---|---:|
| `docs/aneforge-api.md` | 19548 |
| `docs/api/compile.md` | 675 |
| `docs/api/graph.md` | 187 |
| `docs/api/index.md` | 402 |
| `docs/api/math.md` | 393 |
| `docs/api/models.md` | 357 |
| `docs/api/training.md` | 241 |
| `docs/assets/demo.png` | 351518 |
| `docs/assets/demo.tape` | 1599 |
| `docs/assets/fluid_vorticity.webp` | 559682 |
| `docs/assets/game_of_life.webp` | 621570 |
| `docs/assets/lorenz.png` | 115087 |
| `docs/assets/mandelbrot.png` | 113772 |
| `docs/assets/neural_ca.webp` | 196758 |
| `docs/assets/reaction_diffusion.webp` | 562652 |
| `docs/capabilities.json` | 105328 |
| `docs/capabilities.md` | 14128 |
| `docs/cross-chip.md` | 13574 |
| `docs/developer/autograd.md` | 16966 |
| `docs/developer/bridges.md` | 13366 |
| `docs/developer/capabilities-and-targets.md` | 17678 |
| `docs/developer/compile-pipeline.md` | 12996 |
| `docs/developer/models.md` | 13874 |
| `docs/developer/numerics.md` | 17136 |
| `docs/developer/op-quirks.md` | 9611 |
| `docs/developer/optimizer-and-cost.md` | 16199 |
| `docs/developer/overview.md` | 7482 |
| `docs/development.md` | 4178 |
| `docs/dispatch.md` | 7381 |
| `docs/e5rt-dispatch-reference.md` | 16641 |
| `docs/faq.md` | 11226 |
| `docs/gen_op_catalog.py` | 2784 |
| `docs/getting-started.md` | 5868 |
| `docs/glossary.md` | 10254 |
| `docs/index.md` | 3667 |
| `docs/llm.md` | 10764 |
| `docs/mil-primer.md` | 10361 |
| `docs/mlperf.md` | 4514 |
| `docs/numeric-cliffs.md` | 3809 |
| `docs/onnx.md` | 12740 |
| `docs/op-catalog.md` | 19394 |
| `docs/reproducibility.md` | 6415 |
| `docs/requirements.txt` | 312 |
| `docs/roadmap.md` | 18053 |
| `docs/rooflines.md` | 2366 |
| `docs/stylesheets/extra.css` | 535 |
| `docs/training.md` | 18756 |

### mkdocs.yml

| Path | Bytes |
|---|---:|
| `mkdocs.yml` | 3264 |

### pyproject.toml

| Path | Bytes |
|---|---:|
| `pyproject.toml` | 7481 |

### scripts

| Path | Bytes |
|---|---:|
| `scripts/fuzz.py` | 23346 |
| `scripts/fuzz.sh` | 1669 |
| `scripts/reproduce.sh` | 6266 |
| `scripts/sync_hf_dataset.py` | 1554 |

### tests

| Path | Bytes |
|---|---:|
| `tests/_corpus.py` | 5747 |
| `tests/_helpers.py` | 2973 |
| `tests/conftest.py` | 2286 |
| `tests/op_smoketest.py` | 4271 |
| `tests/run_corpus.py` | 1595 |
| `tests/test_ane_rail.py` | 4757 |
| `tests/test_attention_tiles.py` | 1940 |
| `tests/test_autograd.py` | 58780 |
| `tests/test_blas.py` | 7193 |
| `tests/test_broad.py` | 5027 |
| `tests/test_builder_guards.py` | 1275 |
| `tests/test_clip.py` | 4962 |
| `tests/test_compile_breaker.py` | 3319 |
| `tests/test_compile_targets.py` | 3836 |
| `tests/test_compress.py` | 26680 |
| `tests/test_conv_int8.py` | 5098 |
| `tests/test_corners.py` | 6761 |
| `tests/test_cost_model_analytic.py` | 6933 |
| `tests/test_cross_chip_ops.py` | 5067 |
| `tests/test_cross_compile.py` | 2779 |
| `tests/test_cross_compile_matrix.py` | 1820 |
| `tests/test_cross_encoder.py` | 4964 |
| `tests/test_decode_chunking.py` | 2135 |
| `tests/test_decoder_block.py` | 5136 |
| `tests/test_dense_loader_guard.py` | 1656 |
| `tests/test_dispatch_floor_warning.py` | 1532 |
| `tests/test_dsp_detrend.py` | 3617 |
| `tests/test_dsp_guards.py` | 1731 |
| `tests/test_dsp_hilbert.py` | 1688 |
| `tests/test_dsp_selftest.py` | 503 |
| `tests/test_dsp_signal_generators.py` | 2157 |
| `tests/test_dsp_welch.py` | 1194 |
| `tests/test_dsp_windows.py` | 1929 |
| `tests/test_dynamic_conv.py` | 2299 |
| `tests/test_einsum_diagonal.py` | 4398 |
| `tests/test_einsum_ellipsis.py` | 2028 |
| `tests/test_einsum_selftest.py` | 1168 |
| `tests/test_encoder_padding.py` | 1254 |
| `tests/test_espresso_workarounds.py` | 5029 |
| `tests/test_fft.py` | 2199 |
| `tests/test_fft2.py` | 2862 |
| `tests/test_fp16_cross_chip.py` | 4225 |
| `tests/test_fuzz_harness.py` | 3149 |
| `tests/test_game_of_life_example.py` | 1454 |
| `tests/test_gpt2.py` | 10485 |
| `tests/test_group_norm_tiling.py` | 2498 |
| `tests/test_hf_resnet.py` | 3129 |
| `tests/test_image_input.py` | 4356 |
| `tests/test_input_ranges.py` | 4456 |
| `tests/test_lapack.py` | 9231 |
| `tests/test_layer_norm_affine_form.py` | 2352 |
| `tests/test_linalg.py` | 25669 |
| `tests/test_llm.py` | 11957 |
| `tests/test_lm_head.py` | 10050 |
| `tests/test_lorenz_example.py` | 886 |
| `tests/test_machine_dirty_flag.py` | 5242 |
| `tests/test_machine_main_ref.py` | 5085 |
| `tests/test_mandelbrot_example.py` | 1842 |
| `tests/test_mlperf_harness.py` | 6639 |
| `tests/test_moe.py` | 6529 |
| `tests/test_moe_topk_range.py` | 621 |
| `tests/test_mpnet.py` | 2371 |
| `tests/test_multilayer_resident.py` | 569 |
| `tests/test_new_ops.py` | 4108 |
| `tests/test_nn_blocks.py` | 13007 |
| `tests/test_norm_affine_form.py` | 3774 |
| `tests/test_numerical.py` | 15628 |
| `tests/test_onnx.py` | 94481 |
| `tests/test_onnx_axis_attrs.py` | 2297 |
| `tests/test_onnx_coverage.py` | 3019 |
| `tests/test_op_catalog.py` | 1545 |
| `tests/test_op_coverage.py` | 9460 |
| `tests/test_opt1_act_ceiling.py` | 3331 |
| `tests/test_pde_ode.py` | 21118 |
| `tests/test_qwen2_bias.py` | 3707 |
| `tests/test_qwen35.py` | 8304 |
| `tests/test_rag.py` | 2227 |
| `tests/test_resnet_depths.py` | 3582 |
| `tests/test_rewrite_engine.py` | 9904 |
| `tests/test_roberta_encoder.py` | 1317 |
| `tests/test_rope_theta.py` | 1488 |
| `tests/test_routes.py` | 5904 |
| `tests/test_sdpa_causal.py` | 6440 |
| `tests/test_shapes.py` | 5025 |
| `tests/test_special_bessel.py` | 2136 |
| `tests/test_special_digamma_beta.py` | 2737 |
| `tests/test_special_erf_form.py` | 886 |
| `tests/test_special_trig.py` | 4961 |
| `tests/test_spectral_sci.py` | 16810 |
| `tests/test_speculative.py` | 1822 |
| `tests/test_streaming.py` | 3311 |
| `tests/test_synthetic.py` | 8803 |
| `tests/test_targets.py` | 10893 |
| `tests/test_train_cifar.py` | 5536 |
| `tests/test_tune_guards.py` | 8311 |
| `tests/test_variant_gate_failclosed.py` | 1698 |
| `tests/test_vjp_sweep.py` | 2783 |
| `tests/test_whisper.py` | 7103 |
| `tests/test_zero_copy_io.py` | 1955 |

### web

| Path | Bytes |
|---|---:|
| `web/apple-touch-icon.png` | 4952 |
| `web/assets/fluid_vorticity.webp` | 1349664 |
| `web/assets/neural_ca.webp` | 196758 |
| `web/assets/og-card.png` | 170915 |
| `web/assets/reaction_diffusion.webp` | 562652 |
| `web/favicon.ico` | 15086 |
| `web/favicon.png` | 950 |
| `web/favicon.svg` | 325 |
| `web/fonts/OFL.txt` | 4599 |
| `web/fonts/plex-mono-400.woff2` | 10052 |
| `web/fonts/plex-mono-500.woff2` | 10060 |
| `web/fonts/plex-mono-600.woff2` | 10120 |
| `web/fonts/plex-sans-400.woff2` | 40240 |
| `web/fonts/plex-sans-500.woff2` | 40240 |
| `web/fonts/plex-sans-600.woff2` | 40240 |
| `web/index.html` | 27085 |

## Crawl requests that were not pages

A first link walk joined relative URLs one directory too deep. These 20 requests returned HTTP 404 and are not documents on the sites. The pages they were trying to reach were fetched afterward at the paths in the sections above.

- <https://aneforge.readthedocs.io/en/latest/api/math/t>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-1-machine/03-numerics.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-6-silicon/20-datapath-mac.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-4-workloads/13-vision-conv-encoders.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-3-performance/11-ane-vs-gpu-vs-cpu.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-9-cross-silicon/36-upper-tier.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-4-workloads/15-training.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/appendices/c-decoded-tables.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/part-7-toolchain/23-program-format.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/back-matter/statements.html>
- <https://ane-guide.readthedocs.io/en/latest/back-matter/appendices/a-op-device-matrix.html>
- <https://ane-guide.readthedocs.io/en/latest/references.html/appendices/f-provenance.html>
- <https://ane-guide.readthedocs.io/en/latest/appendices/part-2-reaching/07-weights-and-compression.html>
- <https://ane-guide.readthedocs.io/en/latest/appendices/part-7-toolchain/25-compression-internals.html>
- <https://ane-guide.readthedocs.io/en/latest/appendices/f-provenance.html/c-decoded-tables.html>
- <https://ane-guide.readthedocs.io/en/latest/appendices/appendices/d-glossary.html>
- <https://ane-guide.readthedocs.io/en/latest/appendices/references.html>
- <https://ane-guide.readthedocs.io/en/latest/part-1-machine/references.html>
- <https://ane-guide.readthedocs.io/en/latest/part-1-machine/intro.html>
- <https://ane-guide.readthedocs.io/en/latest/part-1-machine/part-1-machine/02-execution-model.html>

## Affected topics

[ANE guide](../guides/ane.md) and [related work](related-work.md) point here. Reading a card above is still required before using a page as evidence.
