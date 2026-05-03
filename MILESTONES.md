# SpQt → MLX: Milestone Plan

Port the SpQt idea from ggml/Metal (llama.cpp-SpQt) to MLX/Metal.

## Goal

Demonstrate that the SpQt zigzag-layout + sparse-GEMV idea transfers from Q4_K-on-ggml
to an MLX-native quantization format, validated by:

- **Correctness:** `(y - y_ref).abs().max() < 1e-3` on a single arbitrary shape
  (matches MLX's quantized-matmul convention at `python/tests/test_quantized.py:355`).
- **Performance:** break-even or better against MLX's stock affine qmv at the same shape and sparsity.
- **Deliverable:** turn-key repo — one command builds the extension and reproduces both gates from a clean clone.

## Working principles

1. **Minimal code, reasonable design.** Write only what the current milestone needs;
   no abstractions for hypothetical future ones; no defensive validation past system
   boundaries. Bias toward fewest moving parts, but don't write code the next
   milestone has to throw away.
2. **User implements; Claude designs and instructs.** From M0b onward, code edits in
   the repo come from the user. Claude's role is design, line-level instruction,
   pointing at references, and review-after — not writing production code unless
   explicitly asked. (Investigation docs in `investigations/` and `MILESTONES.md`
   remain Claude's to maintain.)
3. **Fair comparison, both directions.** Bench against MLX's strongest path
   (`affine_qmv_fast`, fast-aligned shape, no disabled optimizations); apply equal
   best-effort to our kernel — exhaust tuning knobs before declaring M3 failure.
   Apples to apples.

4. **Mirror MLX kernel patterns; diverge only at the algorithmic novelty.** When
   writing Metal kernels, base them on MLX's existing templates (e.g.
   `qmv_fast_impl`) — same template signature `<T, gs, bits>`, same helpers
   (`qdot`, `load_vector`, `get_pack_factor`, `get_bytes_per_pack`), same
   `typedef float U;` accumulator convention, same `simd_sum` reduce idiom.
   Diverge only where SpQt's algorithm genuinely requires it (K-tiled vs.
   row-tiled parallelism, cross-TG atomic-reduce, idx-driven K-walk).
   *Why:* (a) templated portability across bit widths and group sizes is
   nearly free; (b) reusing tested helpers reduces risk on the bit-format
   gymnastics for 3/5/6-bit cases; (c) the kernel reads natively to MLX
   maintainers, which matters for any e2e-integration follow-up.

## Scope decisions

1. **Path 2: MLX-native affine Q4 (group_size=64)**, not a Q4_K port.
   *Why:* Path 1 would require us to also implement the dense Q4_K kernel — the baseline
   we'd be comparing against would then be our own code, making any speedup claim suspect.
   Path 2 compares against MLX's already-tuned `affine_qmv_fast`. The SpQt idea is
   format-agnostic (any per-row, group-along-K asymmetric quantizer), so applying it
   to MLX-native affine Q4 is a faithful transfer.

2. **Sparse-indexing kernel deferred to stretch.** The GEMV consumes a sparse-index
   buffer; for the MVP we build that buffer host-side (`np.flatnonzero`) and upload it.
   *Why:* the GEMV is the centerpiece of the SpQt idea; the index kernel is supporting
   infra (Blelloch prefix scan, well-understood). Document that the paper's end-to-end
   speedup includes index-collection cost that our microbench excludes.

3. **One shape, one bitwidth (Q4), one threshold (≈50% sparsity).** Edge cases, multi-bit,
   GEMM/prefill path, dense fallback, and fused-norm variants are all stretch.

4. **Route B: C++ extension over `mx.fast.metal_kernel`.** Register the kernel as
   a real `Primitive` subclass via the `examples/extensions/axpby` template (pip-
   installable extension with its own metallib).
   *Why:* (a) any e2e-model-inference follow-up would need the extension wrapping
   anyway, so getting the API shape right *now* avoids throw-away work; (b) the
   extension form is the canonical "registered MLX custom operation" shape — kernel
   code reads natively to MLX maintainers, and a future
   `mx.quantized_matmul(mode="affine_zigzag")` would slot directly into this
   structure; (c) the scaffolding is well-templated by axpby — extension overhead
   is bounded (~5-8h on top of the kernel work), not open-ended. The trade-off
   is +25-30% engineering time.

5. **New top-level extension at `extensions/mlx_spqt/`; axpby retained as sanity
   baseline through M0b–M2, dropped at M4.**
   *Why a new extension*, not modifying `examples/extensions/`: keeps our diff
   against upstream MLX purely additive and preserves axpby's role as MLX's
   canonical extension example.
   *Why retain axpby temporarily*: during scaffolding verification, a known-working
   kernel inside our package decouples scaffold-failure modes from kernel-failure
   modes. If `mlx_spqt.axpby(...)` works after the rename, the build / CMake /
   metallib / runtime-path pipeline is sound — independent of any SpQt-specific code.
   *When to drop*: as part of M4 cleanup. Final extension contains only SpQt code.

## Milestones

| #   | Name                                | Status | Done criterion                                                                                          |
| --- | ----------------------------------- | --- | ------------------------------------------------------------------------------------------------------- |
| M0a | Structural map: ggml ↔ MLX           | ✅ | One-page topics × {ggml, MLX, what we use} comparison; in its own doc; every row drives a downstream decision. |
| M0b | Extension scaffold + smoke kernels   | ✅ | `extensions/mlx_spqt/` builds via `pip install -e .`; `mlx_spqt.axpby` (sanity baseline) works after the rename; all four smoke kernels (uint32 buffer, qdot includibility, atomic-add, threadgroup memory) pass. See `investigations/mlx-extension-ours.md` Phase 2. |
| M1  | `zigzag_quantize` + dense zigzag-GEMV Metal kernel | ✅ | `zigzag_quantize(w_fp)` produces zigzag-quantized weights/scales/biases; dense zigzag-GEMV Metal kernel matches `mx.quantized_matmul` to `< 1e-3` on 7 shapes; competitive perf (~1.0× at LLM-FFN, ~1.3× off at smaller). See `investigations/M1-zigzag-layout.md`. |
| M2  | Sparse-GEMV: M1 kernel + idx-driven K-walk | ✅ | Sparse op `zigzag_qmv_sparse` matches reference `< 1e-3` across 7 shapes × 2 densities. Threadgroup geometry tuned via 11-combo × 7-shape × 3-run sweep — default `(NSG=2, TG=4)`. See `investigations/M2-sparse-design.md`. |
| M3  | Performance break-even               | ✅ | **Exceeded** — sparse beats `mx.quantized_matmul` by ~1.8× at 75% sparsity on LLM-FFN shapes; ~1.2× at 50%. Multi-shape × multi-density bench in `bench_zigzag_qmv_sparse.py`. |
| M4  | Turn-key repo                        | ✅ | `python demo_spqt.py` reproduces both correctness + performance from a clean install; top-level `README.md` covers install + run + headline result. |
| S1  | (Stretch) Metal sparse-indexing      | — | Port `kernel_sparse_indexing_v2` (Blelloch scan). Out of scope; idx is host-built for the take-home. |
| S2  | (Stretch) Multi-shape correctness    | ✅ | Tests cover 7 shapes from 1024² to 4096×16384 (Llama-7B FFN sizes). |
| S3  | (Stretch) GEMM/prefill path          | — | Zigzag GEMM analog of `kernel_mul_mm_zigzag`. Out of scope — GEMV-only deliverable. |

---

## Milestone detail

### Exploration

A continuous, parallel process — not on the deliverable critical path. Whenever we
hit something we need to understand more deeply (now or in later milestones), the
output lands in [`investigations/`](investigations/). Long-form reference material;
the focused M0a deliverable distills only the parts that drive M1–M4 decisions.

### M0a — Structural map (focused)

A short, decision-driving comparison. Lives in its own doc (path TBD).

Principles for the deliverable:

- **Format:** single topics × {ggml/llama.cpp, MLX, what we use} table.
- **Scope:** only topics that drive M1–M4 implementation/evaluation choices. No FYI
  rows.
- **Length:** ~1 page; a reader should grasp the port's grounding in 30 seconds.
- **Pairing:** ggml ≈ mlx-core, llama.cpp ≈ mlx.nn — don't compare across layers.

Done criterion: every row maps to a concrete downstream decision. Anything
"interesting but unused" stays in `investigations/`, not in M0a.

### M0b — Extension scaffold + smoke kernels

Per Scope decisions #4 and #5: stand up a new top-level extension at
`extensions/mlx_spqt/` cloned from the axpby template. Keep the axpby kernel
itself as a sanity baseline through development. Verify the build + Metal
features that M2 will rely on.

Reference template: `examples/extensions/` (vanilla axpby — *not* modified).
Walkthrough of every piece: `investigations/mlx-extension-axpby.md`.

**Two artifacts:**

1. **Cloned extension scaffold at `extensions/mlx_spqt/`.** Copy
   `examples/extensions/` (the whole directory) to `extensions/mlx_spqt/`.
   **Keep the `axpby/` subdirectory intact** — per Scope decision #5, axpby
   travels with us as a sanity baseline. Find/replace **package-level**
   identifiers only:
   - `mlx_sample_extensions` → `mlx_spqt` (Python package dir + name)
   - `_ext` → `_spqt_ext` (compiled module name)
   - `mlx_ext` → `mlx_spqt` (metallib `TITLE` and `d.get_library(...)` argument)
   - `mlx_ext_metallib` → `mlx_spqt_metallib` (CMake target name)

   Files touched: `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `bindings.cpp`,
   the renamed Python package's `__init__.py`. **Don't rename `axpby/` itself**
   — it's the verification anchor.

   Verify after `pip install -e .`: `mlx_spqt.axpby(x, y, 2.0, 3.0)` produces
   correct output.

2. **MSL feature smoke kernels** — NEW kernels added alongside `axpby/` in our
   extension. These get their own `.metal` source, Primitive subclasses, and
   bindings. axpby continues to work in parallel.

   - [ ] **Atomic-add into output**: `atomic_fetch_add_explicit` on a `device atomic_float*`
         output. Required for cross-TG row-partial reduce.
   - [ ] **`threadgroup` memory**: cross-simdgroup reduction within a TG using
         `threadgroup` qualifier + `threadgroup_barrier`.
   - [ ] **uint32-packed input**: `mx.array(dtype=mx.uint32)` round-trips through
         a kernel that does nibble extraction (sanity-checks the buffer-binding path
         for our packed weights).
   - [ ] **MLX kernel-helper includibility** (gates Working Principle #4):
         from extension `.metal` source, `#include "mlx/backend/metal/kernels/quantized.h"`
         and call `qdot<float, 16, 4>(...)` on a synthetic input. If this compiles
         and runs, M2 can lean on `qdot` / `load_vector` / `get_pack_factor` directly.
         If it fails, decide vendor-vs-fix-include-paths *now*, before M2 design.

   Each is a ~30-line standalone kernel. If any blocks, M2 design must adapt
   (e.g. avoid cross-TG atomics by using one TG per output row).

**Done criterion**: `pip install -e extensions/mlx_spqt/` succeeds;
`mlx_spqt.axpby(...)` confirms the scaffold; all four smoke kernels pass.

**Effort estimate:** 3-4h scaffold + smoke kernels. Most of the time is renaming
strings in CMake/setup files; the smoke kernels are tiny. Build-environment
issues (Metal toolchain, MLX headers) are the largest tail — fail-fast in the
scaffold step.

### M1 — `zigzag_quantize` + dense zigzag-GEMV kernel

**This is the kernel-design milestone.** M1 builds the full multi-threadgroup,
atomic-reducing kernel architecture for the zigzag layout — but for *dense* input
(walks all K positions). M2 inherits this kernel and changes only the K-walk.
Splitting this way isolates *layout correctness* (M1) from *sparsity correctness*
(M2).

**Two artifacts:**

1. **`zigzag_quantize(w: mx.array, group_size, bits) → (w_zz, scales_zz, biases_zz)`**
   — pure Python/MLX, no Metal. Operates on the **original (un-quantized) fp16/fp32**
   weight matrix; mirrors the signature shape of `mx.quantize(...)`. Internally
   does fp-rearrange-then-quantize (matches `rearrange_tensor_zigzag` at
   `llama.cpp-SpQt/src/llama-quant.cpp:429` which operates on fp32 then quantizes).
   Returns three tensors with dtype/shape conventions analogous to `mx.quantize`.
   This is what a future `mx.quantize(mode="affine_zigzag")` would look like.

2. **Dense zigzag-GEMV Metal kernel** (full kernel; the architectural payload).
   - Templated `<typename T, int group_size, int bits, int num_simdgroups, int TGs_per_band>` from day one. Extends `qmv_fast_impl`'s `<T, gs, bits>` with two M3-tunable knobs.
   - Multi-TG K-tiled architecture with cross-TG atomic-reduce on `out` plus
     within-TG cross-simdgroup reduce via threadgroup memory + barrier.
     Mirrors llama.cpp-SpQt's `kernel_mul_mv_q4_K_f32_zigzag_multi_threadgroup_impl`
     (dense) at `ggml-metal.metal:5702`.
   - Walks contiguous K positions in the zigzag layout (no idx).
   - Reuses MLX kernel helpers per Working Principle #4.

**Design: SpQt-reference threadgroup pattern with qmv_fast structural reuse.**
4 simdgroups × 32 lanes = 128 threads/TG; each lane owns `rows_per_lane =
group_size / SIMD_SIZE` = 2 rows of the band; all SGs cover all 64 rows of the
band, splitting K. Within-TG reduce sums per-row partials across the 4 SGs.
Cross-TG atomic merges TG-level partials. MVP instantiates one specialization
(`<half, 64, 4, 4, 4>`); future tuning adds `(num_simdgroups, TGs_per_band)`
combinations from the same template body. Note: in zigzag, the row-band size
**equals `group_size` by construction** — same as llama.cpp-SpQt where
`superblock_size = QK_K = 256` plays both roles in Q4_K. No separate `M_TILE`
parameter.

| | Reuse from `qmv_fast_impl` | Diverge for SpQt |
|---|---|---|
| Template signature | `<T, group_size, bits>` | + `int num_simdgroups, int TGs_per_band` for M3 tuning |
| Constexpr derivations | `pack_factor`, `bytes_per_pack`, `values_per_thread`, `block_size`, `scale_step_per_thread` — all verbatim | — |
| Activation load | scalar `x[k]` per K-position is sufficient for M1 dense; `load_vector` may return for M2 idx batching | — |
| Accumulator type | `typedef float U; thread U result[rows_per_lane] = {0};` | — |
| Inner dequant + FMA | `qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum)` | **Replace** with custom dequant+FMA loop (see `mlx-qmv-fast.md` §8 for why qdot doesn't fit) |
| Cross-lane reduce | `simd_sum(result[row])` | **Skip** — each lane owns disjoint rows of its SG's accumulator set |
| Outer K-walk | `for (int k = 0; k < in_vec_size; k += block_size)` | **Adapt** — SG walks its `[k_start, k_end)` stripe; **further replace** in M2 with idx-driven walk |
| Output write | Direct `y[row] = result[row]` from lane 0 | **Replace** with two-level reduce: TG-mem + barrier across SGs, then `atomic_fetch_add_explicit` cross-TG |
| Threadgroup geometry | `(SIMD_SIZE, 2, 1)` — 8 rows / TG, all rows on one TG | **Replace** — `(SIMD_SIZE × num_simdgroups, 1, 1)` = 128 threads/TG; SGs share rows, split K; multiple TGs per band |

The "Reuse" rows are M1/M2's non-divergence policy. Header includibility for
`qdot` et al. is gated by M0b smoke test (d).

**Done criterion:**

```python
# w is the original fp16 weight matrix (single shape per Scope decision #3)
w_zz, scales_zz, biases_zz = mlx_spqt.zigzag_quantize(w, group_size=64, bits=4)
y_zz = mlx_spqt.dense_zigzag_qmv(x, w_zz, scales_zz, biases_zz)

# reference: standard (non-zigzag) quantization of the same weights
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
y_ref = mx.quantized_matmul(x, w_q, scales, biases, transpose=True,
                             group_size=64, bits=4)

assert (y_zz - y_ref).abs().max().item() < 1e-3
```

**Why M1 is the heavy lift, not M2.** The K-tiled multi-TG architecture, the
atomic-reduce wiring, the zigzag layout walk, the template parameterization,
the helper-reuse plumbing — all of that lands in M1. M2 changes about 10-20
lines (idx parameter binding + a different inner-loop iterator).

### M2 — Add idx-driven K-walk to M1's kernel

**Incremental on M1.** The kernel architecture (multi-TG, atomic-reduce, template
signature, helper reuse, threadgroup geometry) is established in M1. M2 changes
only how K is walked: instead of contiguous K positions, walk the host-built
non-sparse index list.

Reference for the increment:
`kernel_mul_mv_q4_K_f32_zigzag_multi_threadgroup_sparse_impl` at
`llama.cpp-SpQt/ggml/src/ggml-metal/ggml-metal.metal:5867` — the sparse variant
of M1's reference (`_multi_threadgroup_impl`).

**What changes vs. M1:**

| Element | M1 (dense) | M2 (sparse) |
|---|---|---|
| Inputs | `(x, w_zz, scales_zz, biases_zz)` | + `idx` (`int32` mx.array), `idx_count` (int constant) |
| K-walk | `for (kk = 0; kk < num_k_blocks; kk++) ...` (contiguous) | `for (i = 0; i < idx_count; i += block_size_k) { k = idx[i + ...]; ... }` (idx-driven) |
| Threadgroup geometry | TG owns `group_size` rows × contiguous K-slice | TG owns `group_size` rows × `idx` slice |
| Output write | `atomic_fetch_add_explicit(&y[row], ...)` | unchanged |
| Buffer binding (`eval_gpu`) | 4 input arrays | + `idx` array binding + `idx_count` via `set_bytes` |

That's it. ~10-20 lines of kernel diff against M1, plus the binding update in
`spqt.cpp::eval_gpu`. Most code is shared with M1 either by living in a shared
template or by being a second specialization of the same template body.

**Done criterion** (aligned to MLX convention per
`investigations/mlx-test-bench-infrastructure.md` §1, matching
`test_quantized.py::test_qmv:355`):

```python
# build sparse mask + index host-side
mask = mx.abs(x[0]) > thr
idx = mx.array(np.flatnonzero(np.array(mask)).astype(np.int32))

y_zz = mlx_spqt.zigzag_qmv_sparse(x, w_zz, scales_zz, biases_zz, idx)

# reference: zero out sparse cols of x, then standard MLX qmv on equivalent weights
x_sparse = mx.where(mask, x, mx.array(0.0, dtype=x.dtype))
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
y_ref = mx.quantized_matmul(x_sparse, w_q, scales, biases, transpose=True,
                             group_size=64, bits=4)

assert (y_zz - y_ref).abs().max().item() < 1e-3
```

### M3 — Performance break-even

Microbench shape: M=1, K=4096, N=4096, fp16 activations, gs=64, b=4, ~50% sparsity.
(K%512==0 and N%8==0 → baseline lands on `affine_qmv_fast`, the strongest comparator.)

**Bench methodology** (aligned to MLX convention per
`investigations/mlx-test-bench-infrastructure.md` §2; deeper background in
`investigations/mlx-execution-model.md` §9):

1. **Use the standard `time_fn`** from `benchmarks/python/time_utils.py:8`: 5 warmup
   iterations, 100 timed iterations, **mean ms** per iteration. This matches the
   pattern every existing MLX benchmark uses.
2. **Materialize inputs out of band** before timing: `mx.eval(x, w_q, scales,
   biases, idx)` so the timed region only contains the qmv. (`mx.eval` already
   blocks via the synchronizer event; explicit `mx.synchronize()` is redundant.)
3. **Report two distinct numbers per variant** via the closure-factory pattern (full
   skeleton in `investigations/mlx-test-bench-infrastructure.md` §3):

   ```python
   # single-op closure (one per variant: stock_qmv, spqt_qmv)
   def make_op(...):
       def op():
           return mx.quantized_matmul(...)   # or mlx_spqt.zigzag_qmv(...)
       return op

   # mode wrappers
   def latency_fn(op):
       return op                              # fn returns 1 op → per-call latency

   def throughput_fn(op, n_inner):
       def batched():
           return [op() for _ in range(n_inner)]   # fn returns N un-eval'd ops → batched
       return batched
   ```

   | Mode | `fn` shape | What it measures |
   | --- | --- | --- |
   | **Per-call latency** | `fn` returns one op result | one kernel per command buffer per `time_fn` iter; encode+commit overhead included per kernel |
   | **Batched throughput** | `fn` returns `[op() for _ in range(N_INNER)]` un-eval'd | `time_fn`'s outer `mx.eval` batches N ops into shared command buffers; per-op time = reported msec ÷ N_INNER |

   Pick `N_INNER ≥ 50` (≥ `max_ops_per_buffer` on M-series so the batched regime
   is reached). Report both per variant, so the final table is 4 numbers:

   |  | Stock `affine_qmv_fast` | SpQt-MLX zigzag |
   |---|---|---|
   | per-call (msec/op) | … | … |
   | throughput (msec/op @ N=50) | … | … |

4. **Cross-check on device.** Use `mx.metal.start_capture` + Xcode Instruments to
   confirm the timed region is dominated by the kernel and not by Python/encode/commit.
   Useful evidence that the wall-clock numbers reflect kernel time, not framework overhead.

**Variants compared:**

| Variant                       | What's measured                                  |
| ----------------------------- | ------------------------------------------------ |
| MLX stock `affine_qmv_fast`   | Dense GEMV on original packed weights.           |
| SpQt-MLX zigzag sparse        | This port. Index buffer pre-built, **not timed.** |

**Pass:** SpQt-MLX ≥ stock at 50% sparsity in **both** modes (per-call AND batched).

**If M3 fails:** profile with Xcode Instruments, iterate on the n_TG_K /
simdgroups-per-TG / row-tile size knobs. Don't chase the paper's 1.55× — break-even
is the bar.

### M4 — Turn-key repo

- `README.md`: install + one-command repro of both gates.
- `tests/test_spqt.py`: M2 correctness + M3 perf, prints pass/fail.
- **Drop the `axpby/` sanity baseline from `extensions/mlx_spqt/`.** Per Scope
  decision #5, axpby travels with us through M0b–M2 as a verification anchor;
  M4 is when the final extension gets cleaned to contain only SpQt code. Remove
  the `axpby/` subdirectory, its `bindings.cpp` registration, and its CMakeLists
  entries.
- Clean diff between `main` (vanilla upstream MLX) and `rxpwang/spqt` — the diff
  *is* the deliverable; should tell the porting story end-to-end.

---

## Risks

1. **`affine_qmv_fast` is well-tuned.** Beating it at 50% sparsity is not free. M3 is
   the real risk, not M2.
2. **Cross-threadgroup atomics on `out`.** Worked for them on Q4_K; need to confirm
   contention is tolerable at our threadgroup count. Tuning knob.
3. **Extension build environment.** The axpby template requires CMake 3.27+,
   nanobind, MLX with `MLX_BUILD_METAL=ON`, Python dev headers. Verify the
   toolchain in M0b before committing kernel-design time. The `mlx_build_metallib`
   CMake macro and `current_binary_dir()`/`d.get_library(...)` runtime path
   resolution are the most fragile pieces — prove them out with the trivial kernel
   in M0b.
4. **Layout choice (zigzag block size).** Paper uses 256 (= one Q4_K super-block).
   MLX-affine has 64-element groups. Picking 64 keeps scale lookup trivial; picking
   256 (= 4 groups) better matches the paper's tiling. Start with 64, revisit if M3 fails.
5. **Pipeline-state caching skews first-iteration timing.** Always warm up; the first
   eval after a kernel build pays metallib load + JIT.
6. **Command-buffer batching skews per-call vs. batched timing silently.** Eval inside
   the loop = one kernel per buffer (per-call). Eval once after building a list =
   shared buffer (batched). Mixing the two modes invalidates comparisons. Pick one
   per measurement, document explicitly.
7. **Time pressure on stretch.** Resist scope creep. M2 + M3 + M4 ≫ S1.

## Reporting honesty

What the README, bench output, and any documentation should make explicit:

- **Index buffer is host-built and pre-uploaded.** Index-collection cost is not in
  our timings. The paper's 1.55× end-to-end speedup includes it; our microbench
  doesn't.
- **Single shape only.** Not a sweep. Not a full forward pass.
- **Report both per-call latency and batched throughput**, explaining the
  difference (command-buffer batching). Don't quote one number without saying which.
- **Numbers are backed by on-device evidence.** Xcode Instruments / Metal trace
  capture confirms wall-clock timing is dominated by the kernel, not Python /
  encode / commit overhead.
- **Scope of the validation.** What's been validated is "the zigzag-GEMV idea
  transfers to a different group-quantization format and remains performance-
  competitive", not "production-ready SpQt-MLX".

---

## Appendix: where to look things up

The deep structural map originally planned for this appendix lives in
`investigations/mlx-affine-quantization.md` and `investigations/mlx-execution-model.md`.
Quick pointers:

| Topic | See |
| --- | --- |
| Test patterns for quantized ops (`test_quantized.py`) | `mlx-test-bench-infrastructure.md` §1 |
| `mx.fast.metal_kernel` API surface (covered for reference, not used) | `mlx-test-bench-infrastructure.md` §1 |
| Bench utilities (`time_fn`, `gather_qmm_bench.py`) | `mlx-test-bench-infrastructure.md` §2 |
| Skeleton code for our M2/M3 calls | `mlx-test-bench-infrastructure.md` §3 |
| MLX quantization modes (affine, mxfp4, nvfp4, mxfp8) | `mlx-affine-quantization.md` §1 |
| Affine origin, math, popularity | `mlx-affine-quantization.md` §2 |
| Memory layout (packed_w, scales, biases shapes) | `mlx-affine-quantization.md` §3 |
| Comparison vs. ggml Q4_K | `mlx-affine-quantization.md` §4 |
| Full kernel landscape (~30 templates) | `mlx-affine-quantization.md` §5 |
| GEMV-only subset (`qmv_fast` / `qmv` / `qmv_quad`) | `mlx-affine-quantization.md` §6 |
| The exact baseline kernel for our shape | `mlx-affine-quantization.md` §7 |
| Lazy graph (no graph object, implicit DAG) | `mlx-execution-model.md` §1 |
| Op-call → primitive node construction | `mlx-execution-model.md` §2 |
| `eval` 3-phase algorithm (DFS / tape / dispatch) | `mlx-execution-model.md` §3 |
| `gpu::eval` encode + commit detail | `mlx-execution-model.md` §4 |
| Metal layer cake (Device → Queue → Buffer → Encoder) | `mlx-execution-model.md` §5 |
| Single-stream behavior in our scenario | `mlx-execution-model.md` §6 |
| Synchronizer pattern (how `eval` blocks) | `mlx-execution-model.md` §8 |
| Bench methodology implications | `mlx-execution-model.md` §9 |
