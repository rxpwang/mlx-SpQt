# mlx-SpQt — SpQt port to MLX

This repository is a fork of [`ml-explore/mlx`](https://github.com/ml-explore/mlx) implementing the [SpQt](https://arxiv.org/abs/2511.04477) port to MLX:

> **SpQt** ("Sparse-Promoted Quantization with zigzag layout") is a runtime-sparse quantized GEMV scheme that rearranges 4-bit weights such that selected K-positions can be **skipped** rather than masked, saving memory bandwidth proportionally to sparsity.

**Two-phase approach.** We first built the kernel as a standalone MLX *extension* under `extensions/mlx_spqt/` — letting us validate kernel design, threadgroup tuning, and multi-shape correctness in isolation, without touching MLX core. Then we *integrated* the validated kernel into MLX core itself, surfacing it as `mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse` (the production deliverable). Integrating an already-tested extension was substantially cheaper than going direct: the kernel and Primitive structure were already proven, leaving only the core's API plumbing to wire up.

---

## What's in here

Two parallel deliverables — same kernel, two API surfaces:

### Integrated MLX core ops (production)

```python
import mlx.core as mx
from mlx.spqt import quantize_zigzag, dequantize_zigzag

w_q, scales, biases = quantize_zigzag(w, group_size=64, bits=4)
y = mx.zigzag_qmv_dense(x, w_q, scales, biases)
y = mx.zigzag_qmv_sparse(x, sparse_indices, w_q, scales, biases)
```

| Op | Layer | What |
|---|---|---|
| `mx.zigzag_qmv_dense` | `mlx.core` (C++ Primitive) | Same logical math as `mx.quantized_matmul` but on zigzag layout. |
| `mx.zigzag_qmv_sparse` | `mlx.core` (C++ Primitive) | Skips K-positions not in `sparse_indices` — kernel time scales with `\|idx\|`, not `K`. |
| `mlx.spqt.quantize_zigzag` | `mlx.spqt` (Python helper) | fp-rearrange + `mx.quantize` round-trip. |
| `mlx.spqt.dequantize_zigzag` | `mlx.spqt` (Python helper) | Symmetric inverse. |

### Extension `mlx_spqt` (dev/validation scaffold)

The same ops are also exposed under `mlx_spqt.*` in the standalone extension under `extensions/mlx_spqt/`. The extension was Phase 1 — built first to de-risk the kernel design before touching MLX core. It's preserved as the parallel reference (and hosts additional artifacts like the threadgroup-tuning sweep, M0b smoke kernels, and the unified demo script).

The sparse op exposes `num_simdgroups` and `threadgroups_per_band` as runtime kwargs (defaulting to `(2, 4)` per a multi-run threadgroup-geometry sweep — see `investigations/M2-sparse-design.md` §5). Power users can override for shape-specific tuning.

---

## Headline result

Sparse zigzag-GEMV vs MLX's tuned `mx.quantized_matmul`, per-call μs at LOOPS=32, threadgroup geometry default `(num_simdgroups=2, threadgroups_per_band=4)`. **Test hardware: MacBook Pro with M4 Pro.** Absolute timings are hardware-dependent; speedup ratios should be broadly reproducible across M-series chips, with absolute numbers scaling with each chip's memory bandwidth.

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

> **Metal Toolchain prerequisite (Xcode 16+ / macOS 15+).** Recent Xcode releases split the Metal compiler into a separately-downloadable component. If `pip install -e .` fails with `cannot execute tool 'metal' due to missing Metal Toolchain`, install it once:
> ```bash
> xcodebuild -downloadComponent MetalToolchain
> ```
> Verify with `xcrun -sdk macosx metal --version`, then retry the install.

> **Branch matters.** The integrated `mx.zigzag_qmv_*` APIs and the Phase 2 artifacts (`mlx/backend/metal/quantized_zigzag.*`, `mlx.spqt`, `python/tests/test_quantized_zigzag.py`, `benchmarks/python/quantized_zigzag_bench.py`, top-level `demo_spqt.py`) live on the **`rxpwang/spqt-integrated`** branch. The earlier `rxpwang/spqt` branch contains only the Phase 1 extension. Make sure you're on the right branch before building.

```bash
git clone https://github.com/rxpwang/mlx-SpQt.git
cd mlx-SpQt
git checkout rxpwang/spqt-integrated     # Phase 2 (integrated MLX core ops + extension)
pip install -e .                          # ~5-10 minutes; builds MLX core including the integrated zigzag kernels
```

That's it. The integrated APIs are now reachable from anywhere:

```python
import mlx.core as mx
from mlx.spqt import quantize_zigzag, dequantize_zigzag
print(mx.zigzag_qmv_dense, mx.zigzag_qmv_sparse)
```

The Phase 1 extension (`mlx_spqt.*`) is *optional* — only needed if you want to run the extension-version scripts under `extensions/mlx_spqt/` (see "Phase 1 extension scripts" below).

### Quickstart — single-script demo

The shortest path to "this works":

```bash
python demo_spqt.py
```

End-to-end demo of the integrated APIs: correctness AND performance for both `mx.zigzag_qmv_dense` and `mx.zigzag_qmv_sparse`, across 7 shapes × 4 sparsity densities. Total runtime ~1.5-2 minutes. Output has three sections:

1. **Correctness** — table of max abs errors vs fp32 reference (35 cases, threshold 1e-3).
2. **Performance** — per-shape per-call timings: `mx.zigzag_qmv_dense`, `mx.quantized_matmul`, naive `x @ w_dq.T`, and sparse at each density.
3. **Summary** — headline speedup numbers (sparse@density=0.25 vs `mx.quantized_matmul`).

A single Python script that exercises both correctness and performance from MLX's public Python API — generalized to multi-shape × multi-density coverage.

### Tests and bench (integrated)

```bash
# Correctness — pytest under MLX's test layout
cd python/tests && python -m unittest test_quantized_zigzag -v

# Performance bench — multi-shape × multi-density
python benchmarks/python/quantized_zigzag_bench.py
```

### Phase 1 extension scripts (optional)

The Phase 1 extension under `extensions/mlx_spqt/` exposes the same kernel via `mlx_spqt.*` and ships parallel demo / test / bench scripts plus the threadgroup-tuning sweep. To enable it, build the extension on top of MLX core:

```bash
cd extensions/mlx_spqt
pip install -e .
```

Then:

```bash
python demo_spqt.py                   # end-to-end demo (correctness + perf)
python test_zigzag_qmv_dense.py       # 7 shapes, dense GEMV vs fp32 reference
python test_zigzag_qmv_sparse.py      # 7 shapes × 2 densities, sparse vs fp32 reference
python bench_zigzag_qmv_dense.py      # dense vs mx.quantized_matmul + naive
python bench_zigzag_qmv_sparse.py     # 4 densities for sparse, vs the same baselines
python sweep_sparse_geometry.py       # 11-combo × 7-shape × 3-run threadgroup sweep
```

---

## Project layout

```
.
├── demo_spqt.py                          ← Phase 2 end-to-end demo (integrated APIs)
│
├── mlx/                                  ← MLX core (forked); Phase 2 integration lives here
│   ├── backend/metal/quantized_zigzag.cpp        ← Phase 2: Primitive eval_gpu
│   ├── backend/metal/kernels/quantized_zigzag.{h,metal}  ← Phase 2: kernel templates + instantiations
│   ├── ops.{h,cpp}                       ← Phase 2: public C++ API mx::zigzag_qmv_*
│   └── primitives.{h,cpp}                ← Phase 2: ZigzagQmvDense / ZigzagQmvSparse classes
│
├── python/                               ← MLX Python bindings; Phase 2 surface
│   ├── mlx/spqt.py                       ← Phase 2: mlx.spqt.quantize_zigzag/dequantize_zigzag
│   ├── src/ops.cpp                       ← Phase 2: nanobind binding for mx.zigzag_qmv_*
│   └── tests/test_quantized_zigzag.py    ← Phase 2: pytest correctness suite
│
├── benchmarks/python/quantized_zigzag_bench.py   ← Phase 2: perf bench
│
├── extensions/mlx_spqt/                  ← Phase 1: standalone MLX extension (dev/validation scaffold)
│   ├── mlx_spqt/                         ← Python package (quantize_zigzag, dequantize_zigzag)
│   ├── zigzag_qmv_dense/                 ← Dense GEMV custom op (M1) — kernel + Primitive + Metal
│   ├── zigzag_qmv_sparse/                ← Sparse GEMV custom op (M2) — same shape, sparse K-walk
│   ├── bindings.cpp, CMakeLists.txt      ← Extension build + bindings
│   ├── demo_spqt.py                      ← Phase 1 end-to-end demo (extension APIs)
│   ├── sweep_sparse_geometry.py          ← Threadgroup-geometry tuning sweep (11 × 7 × 3 = 462 measurements)
│   ├── test_*.py                         ← Per-op correctness
│   └── bench_*.py                        ← Per-op throughput
│
├── investigations/                       ← Design docs
│   ├── M1-zigzag-layout.md               ← Phase 1: dense kernel design + lessons
│   ├── M2-sparse-design.md               ← Phase 1: sparse kernel diff + bench analysis + tuning sweep
│   ├── I0-mlx-core-integration-map.md    ← Phase 2: integration structural map
│   ├── I1..I5-design.md                  ← Phase 2: per-phase design notes
│   ├── mlx-spqt-reference.md             ← Walkthrough of llama.cpp-SpQt's kernel
│   ├── mlx-qmv-fast.md                   ← Walkthrough of MLX's qmv_fast_impl
│   └── ...
│
├── MILESTONES.md                         ← Project plan + working principles + status
└── README.md                             ← (this file)
```

---

## Approach

The work was structured into staggered milestones across two phases:

| | Goal | Status |
|---|---|---|
| **M0a** | Understand MLX-affine quantization layout, `qmv_fast` kernel structure, MLX extension mechanism | ✓ |
| **M0b** | Smoke kernels verifying 4 MSL features needed for the eventual main kernel: uint32 nibble unpack, qdot/load_vector availability, cross-TG `atomic<float>`, threadgroup memory + barrier | ✓ |
| **M1** | Dense zigzag-GEMV registered as MLX Custom Op; multi-shape correctness; benchmarked against `mx.quantized_matmul` | ✓ |
| **M2** | Sparse zigzag-GEMV (idx-driven K-walk); demonstrate the SpQt skip-K-positions value proposition | ✓ |
| **I0–I5** | Integrate the validated extension into MLX core: kernel files, Primitives, op functions, Python bindings, tests, perf bench. Headline ~1.8× speedup reproduces. | ✓ |

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

- **Phase 1 (extension)**: kernel built and validated under `extensions/mlx_spqt/` — multi-shape correctness across 7 LLM-relevant shapes (1024² up to 4096×16384), 2 densities for sparse (1.0, 0.5). Multi-shape × multi-density bench (4 densities for sparse). Threadgroup-geometry tuning sweep.
- **Phase 2 (integration)**: validated kernel ported into MLX core. `mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse` and `mlx.spqt.quantize_zigzag` / `dequantize_zigzag` callable from any MLX-using Python program. Headline ~1.8× speedup at LLM-FFN shapes reproduces within 5% of the extension version.

What's deferred (with rough effort estimates):

| Item | Why deferred | Estimated effort |
|---|---|---|
| **Coarse tail handling** (densities like 0.6, 0.8 where `n_total % 16 ≠ 0`) | MVP targets densities where `n_total % 16 == 0`; arbitrary-density support is a polish item | ~30-45 min kernel work |
| **Multi-batch (B > 1)** | Locked to B=1 for MVP | ~1-2 hours |
| **Within-SG `simd_sum` cross-lane reduce** | Memory-bound at production shapes; modest expected gain | ~3-4 hours |
| **CPU fallback** | Metal-only is sufficient for the deliverable | ~1-2 days |
| **Multi-platform integration** (CPU + CUDA backends in MLX core) | Metal-only is sufficient for the deliverable; multi-platform unlocks the full upstream PR cycle | +6–10 days kernel work, weeks PR review |
| **JIT-source kernel registration** in MLX core | NoJIT path is the default and is wired up; JIT path is a polish item | ~1-2 hours |

For per-phase integration design docs, see `investigations/I0-I5-*.md`. For the original extension milestones, see `MILESTONES.md`.

---

## Acknowledgments

- Built on top of [Apple's MLX framework](https://github.com/ml-explore/mlx) (`ml-explore/mlx`).
- The SpQt algorithm is from [Wang et al., "SpQt: Sparse-Promoted Quantization with zigzag layout"](https://arxiv.org/abs/2511.04477).
- Reference C++/Metal implementation in `llama.cpp-SpQt` provided the structural template for our zigzag layout + multi-threadgroup kernel pattern.

---

## License

Inherits MLX's MIT license. SpQt code (the integration in `mlx/` / `python/` and the standalone extension in `extensions/mlx_spqt/`) is provided under the same terms.

The upstream MLX README is preserved at [`README_MLX.md`](README_MLX.md).
