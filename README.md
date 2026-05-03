# mlx-SpQt — SpQt port to MLX

This repository is a fork of [`ml-explore/mlx`](https://github.com/ml-explore/mlx) with a custom extension under `extensions/mlx_spqt/` that implements the [SpQt](https://arxiv.org/abs/2511.04477) port to MLX:

> **SpQt** ("Sparse-Promoted Quantization with zigzag layout") is a runtime-sparse quantized GEMV scheme that rearranges 4-bit weights such that selected K-positions can be **skipped** rather than masked, saving memory bandwidth proportionally to sparsity.

This work was done as a take-home project for the **Argmax MTS interview** (May 2026). The deliverable: registered MLX Custom Operations for both dense and sparse zigzag-quantized matrix-vector multiply, multi-shape correctness, and multi-shape × multi-density performance bench.

---

## What's in here

Two MLX Custom Operations registered under `mlx_spqt`:

| Op | Computes | Key property |
|---|---|---|
| `mlx_spqt.zigzag_qmv_dense` | `y = x @ dequantize(W_zz).T` | Same logical math as `mx.quantized_matmul` but on zigzag layout. |
| `mlx_spqt.zigzag_qmv_sparse` | `y = x[:, idx] @ dequantize(W_zz)[:, idx].T` | Skips K-positions not in `idx` — kernel time scales with `\|idx\|`, not `K`. |

Plus Python helpers for the layout transform:

- `mlx_spqt.quantize_zigzag(w, group_size, bits)` — fp-rearrange + quantize. Mirrors `mx.quantize`'s signature shape.
- `mlx_spqt.dequantize_zigzag(w_zz, scales, biases, ...)` — symmetric inverse.

All four are documented and tested across 7 LLM-relevant shapes × 4 sparsity densities.

The sparse op also exposes `num_simdgroups` and `threadgroups_per_band` as runtime kwargs (defaulting to `(2, 4)` per a multi-run threadgroup-geometry sweep — see `investigations/M2-sparse-design.md` §5). Power users can override for shape-specific tuning.

---

## Headline result

Sparse zigzag-GEMV vs MLX's tuned `mx.quantized_matmul`, per-call μs at LOOPS=32, threadgroup geometry default `(num_simdgroups=2, threadgroups_per_band=4)`:

| Shape | weights | mlx_qmv (full) | sparse@0.50 | speedup@0.50 | sparse@0.25 | **speedup@0.25** |
|---|---|---|---|---|---|---|
| (1024, 1024) | 0.5 MB | 11 | 13 | 0.85× | 12 | 0.92× |
| (2048, 2048) | 2 MB | 15 | 17 | 0.88× | 19 | 0.79× |
| (4096, 4096) | 8 MB | 34 | 34 | 1.00× | 25 | **1.36×** |
| (8192, 8192) | 32 MB | 169 | 140 | 1.21× | 102 | **1.66×** |
| (4096, 11008) — Llama-7B FFN-down | 22 MB | 118 | 95 | 1.24× | 62 | **1.90×** |
| (11008, 4096) — Llama-7B FFN-up | 22 MB | 113 | 100 | 1.13× | 62 | **1.82×** |
| (4096, 16384) | 32 MB | 171 | 138 | 1.24× | 97 | **1.76×** |

**Two regimes** are clearly visible:
- **Overhead-bound (≤2 MB weights)**: sparse loses to `mx.quantized_matmul` because per-dispatch overhead dominates. Skipping K-positions doesn't help when the K-walk is already short.
- **Memory-bound (≥8 MB weights)**: sparse wins. At LLM-FFN shapes (22-32 MB), the speedup at 75% sparsity averages **~1.8×** — the SpQt value proposition realized: skipping inactive K-positions saves memory bandwidth proportionally.

For the full multi-shape × multi-density bench data and the threadgroup tuning sweep, see `investigations/M2-sparse-design.md` §4 and §5.

---

## Installation

You need an Apple Silicon Mac (M1 or later — needed for `atomic<float>` and `atomic<int>` Metal features) and Python 3.8+.

### From source (recommended for the take-home review)

This repo bundles a fork of MLX core that needs to be built first:

```bash
# 1. Clone (the MLX-core fork is in-tree, no submodules needed)
git clone https://github.com/rxpwang/mlx-SpQt.git
cd mlx-SpQt

# 2. Build the MLX-core fork (one-time, ~5-10 minutes)
pip install -e .

# 3. Build the SpQt extension on top of MLX
cd extensions/mlx_spqt
pip install -e .
```

After that, `mlx_spqt` is importable from anywhere:

```python
import mlx.core as mx
import mlx_spqt
print(mlx_spqt.zigzag_qmv_dense, mlx_spqt.zigzag_qmv_sparse)
```

### Quickstart — single-script demo

The shortest path to "this works":

```bash
cd extensions/mlx_spqt
python demo_spqt.py
```

This is the unified end-to-end demo that runs both correctness AND performance for both `zigzag_qmv_dense` and `zigzag_qmv_sparse`, across 7 shapes × 4 sparsity densities. Total runtime ~1.5-2 minutes. Output has three sections:

1. **Correctness** — table of max abs errors vs fp32 reference (35 cases, threshold 1e-3).
2. **Performance** — per-shape per-call timings: zigzag dense, `mx.quantized_matmul`, naive `x @ w_dq.T`, and sparse at each density.
3. **Summary** — headline speedup numbers (sparse@density=0.25 vs `mx.quantized_matmul`).

This is the script that satisfies the take-home brief's *"Python script that runs the single-shape correctness and performance-breakeven tests with MLX Python APIs"* — generalized to multi-shape × multi-density coverage.

### More granular tests and benches

The demo wraps these individual scripts; run any of them standalone for narrower focus:

```bash
cd extensions/mlx_spqt

# Correctness
python test_quantize_zigzag.py        # round-trip quantize <-> dequantize
python test_zigzag_qmv_dense.py       # 7 shapes, dense GEMV vs fp32 reference
python test_zigzag_qmv_sparse.py      # 7 shapes × 2 densities, sparse vs fp32 reference

# Performance
python bench_zigzag_qmv_dense.py      # 7 shapes, dense vs mx.quantized_matmul + naive
python bench_zigzag_qmv_sparse.py     # 7 shapes × 4 densities, sparse vs the same baselines
```

---

## Project layout

```
.
├── extensions/mlx_spqt/                  ← The SpQt MLX extension
│   ├── mlx_spqt/                         ← Python package
│   │   ├── __init__.py                   ← Re-exports for both ops + helpers
│   │   └── quantization.py               ← quantize_zigzag, dequantize_zigzag
│   ├── zigzag_qmv_dense/                 ← Dense GEMV custom op (M1)
│   │   ├── zigzag_qmv_dense.{h,cpp}      ← Primitive + op function
│   │   └── zigzag_qmv_dense.metal        ← Templated Metal kernel
│   ├── zigzag_qmv_sparse/                ← Sparse GEMV custom op (M2)
│   │   └── zigzag_qmv_sparse.{h,cpp,metal}
│   ├── bindings.cpp                      ← nanobind Python bindings for the ops
│   ├── CMakeLists.txt                    ← Extension build
│   ├── demo_spqt.py                      ← End-to-end demo (correctness + perf, all shapes/densities)
│   ├── sweep_sparse_geometry.py          ← Threadgroup-geometry tuning sweep
│   ├── test_*.py                         ← Multi-shape correctness (per-op)
│   └── bench_*.py                        ← Multi-shape × multi-density throughput (per-op)
│
├── investigations/                       ← Design docs + technical investigations
│   ├── M1-zigzag-layout.md               ← Dense kernel design + as-built + lessons
│   ├── M2-sparse-design.md               ← Sparse kernel diff + bench analysis
│   ├── mlx-spqt-reference.md             ← Walkthrough of llama.cpp-SpQt's kernel
│   ├── mlx-qmv-fast.md                   ← Walkthrough of MLX's qmv_fast_impl
│   ├── mlx-extension-axpby.md            ← MLX extension mechanism deep-dive
│   ├── mlx-extension-ours.md             ← Living doc tracking the extension's evolution
│   └── ...
│
├── MILESTONES.md                         ← Project plan + working principles + status
├── mlx/                                  ← Upstream MLX source (forked)
└── README.md                             ← (this file)
```

---

## Approach

The work was structured into staggered milestones:

| | Goal | Status |
|---|---|---|
| **M0a** | Understand MLX-affine quantization layout, `qmv_fast` kernel structure, MLX extension mechanism | ✓ |
| **M0b** | Smoke kernels verifying 4 MSL features needed for the eventual main kernel: uint32 nibble unpack, qdot/load_vector availability, cross-TG `atomic<float>`, threadgroup memory + barrier | ✓ |
| **M1** | Dense zigzag-GEMV registered as MLX Custom Op; multi-shape correctness; benchmarked against `mx.quantized_matmul` | ✓ |
| **M2** | Sparse zigzag-GEMV (idx-driven K-walk); demonstrate the SpQt skip-K-positions value proposition | ✓ |

See `MILESTONES.md` for the full plan with working principles, scope decisions, and remaining stretch items.

### Key design choices

- **Path 2** (MLX-native affine 4-bit) over Path 1 (port Q4_K). Reason: gives a fair comparison against `mx.quantized_matmul`, which is MLX's tuned baseline. Q4_K port would need a separate Q4_K → MLX-affine layout converter.
- **Zigzag rearrangement = `group_size` row-tile**: the natural quantization unit IS the row-band size in zigzag. We dropped the M_TILE parameter (originally a separate template knob) once we confirmed the layout makes them inherently equal — same simplification llama.cpp-SpQt's reference exploits.
- **Threadgroup atomic<int> + SCALE_FACTOR**: float threadgroup atomics are unsupported in MSL. We use the int+scaling workaround (canonical pattern in llama.cpp-SpQt) to enable within-TG reduction at production speed, then convert back to float for the cross-TG atomic-add.
- **Custom inner loop, not `qdot` reuse**: zigzag's "16 weights → 16 row-partials per K-pos" inverts qmv_fast's "16 weights × 16 K-pos → 1 partial." `qdot`'s assumed shape doesn't fit; we wrote the per-K-pos dequant+FMA loop directly.
- **Threadgroup geometry default `(NSG=2, TG=4)`** for sparse: chosen via an 11-combo × 7-shape × 3-run sweep (`sweep_sparse_geometry.py`). Within 7% of per-cell best across all 42 measurements; single rule beats shape-dependent dispatch when the per-shape variance is at run-to-run noise level.

For full implementation reasoning + lessons learned (incl. several false starts), see `investigations/M1-zigzag-layout.md` §11-12 and `investigations/M2-sparse-design.md` §5-7.

---

## Limitations and future work

What we deliver:

- Multi-shape correctness across 7 LLM-relevant shapes (1024² up to 4096×16384), 2 densities for sparse (1.0, 0.5).
- Multi-shape × multi-density bench (4 densities for sparse).
- Beats `mx.quantized_matmul` at production sparsities (~1.7× at 75% sparsity).

What's deferred (with rough effort estimates):

| Item | Why deferred | Estimated effort |
|---|---|---|
| **Coarse tail handling** (densities like 0.6, 0.8 where `n_total % 16 ≠ 0`) | Take-home spec only requires "single arbitrary shape" | ~30-45 min kernel work |
| **Multi-batch (B > 1)** | Locked to B=1 for MVP | ~1-2 hours |
| **Within-SG `simd_sum` cross-lane reduce** | Memory-bound at production shapes; modest expected gain | ~3-4 hours |
| **CPU fallback** | Metal-only is sufficient for take-home | ~1-2 days |
| **Upstream merge into MLX core** | Beyond take-home scope | ~2 days for Metal-only standalone integration; weeks for full multi-platform PR cycle |

For details on the upstream-merge path, see the take-home brief context in `MILESTONES.md` and `investigations/M2-sparse-design.md` §5.

---

## Acknowledgments

- Built on top of [Apple's MLX framework](https://github.com/ml-explore/mlx) (`ml-explore/mlx`).
- The SpQt algorithm is from [Wang et al., "SpQt: Sparse-Promoted Quantization with zigzag layout"](https://arxiv.org/abs/2511.04477) (which the candidate is first author of).
- Reference C++/Metal implementation in `llama.cpp-SpQt` provided the structural template for our zigzag layout + multi-threadgroup kernel pattern.

---

## License

Inherits MLX's MIT license. SpQt extension code in `extensions/mlx_spqt/` is provided under the same terms.

The upstream MLX README is preserved at [`README_MLX.md`](README_MLX.md).
