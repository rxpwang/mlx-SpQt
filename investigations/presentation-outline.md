# Presentation outline

A 30-minute talk on the SpQt → MLX port: what was built, why those
choices, and where the result leaves performance.

## Audience

Highly technical, but not domain-experts on Q4_K, MLX-affine, or
zigzag layout specifics. Slides should rely on diagrams over MSL
code, and on concrete numbers (~1.8× speedup, 22 MB FFN matrix)
over abstract claims.

## Time budget

~25 min speaking + 5 min Q&A. With nine sections, ~3 min/section
average; M1 and M2 weighted heavier.

| Time | Section |
|---|---|
| 0:00–0:30 | Title |
| 0:30–4:30 | 1. SpQt idea (4 min) |
| 4:30–6:30 | 2. Take-home scope (2 min) |
| 6:30–7:30 | 3. Roadmap (1 min) |
| 7:30–10:30 | 4. M0 — foundation (3 min) |
| 10:30–14:00 | 5. M1 — dense (3.5 min) |
| 14:00–19:30 | 6. M2 — sparse + tuning (5.5 min) |
| 19:30–22:30 | 7. Results (3 min) |
| 22:30–24:30 | 8. Future work + integration (2 min) |
| 24:30–25:00 | 9. Closing (0.5 min) |
| 25:00–30:00 | Q&A |

## Section 1 — SpQt idea (4 min)

Reuse paper figures.

- Bandwidth wall: ~200 GB/s on M-series; 22 MB FFN matrix.
- Activations are runtime-sparse; naive masking still loads all
  weights.
- Zigzag rearrangement: inactive K-positions become contiguous
  blocks to skip rather than scattered loads to mask.

## Section 2 — Take-home scope (2 min)

Two paths offered:
- Path 1: port the existing SpQt Metal kernel + Q4_K quantization.
- Path 2: port the *idea* using MLX-native quantization.

Chose Path 2. It enables a fair comparison against
`mx.quantized_matmul` (MLX's tuned baseline). Path 1 would have
required a Q4_K-to-MLX-affine layout converter and would compare
against a less-tuned Q4_K path.

## Section 3 — Roadmap (1 min)

MILESTONES.md as a visual.

- Staggered: M0 unblocks M1; M1's kernel structure is reused in
  M2 with only the K-walk changing (~10-line diff).
- MVP focus: coarse-tail handling, multi-batch, and simd_sum
  reduce are deferred.
- Working principles: minimal code, fair comparison.

## Section 4 — M0: foundation (3 min)

Four smoke kernels built before the main kernel; ~half a day of
front-loaded verification.

- `smoke_uint32`: uint32-packed nibble extraction on Metal.
- `smoke_qdot`: MLX's `qdot` / `load_vector` callable from an
  extension.
- `smoke_atomic`: cross-TG `atomic<float>` lossless across 256
  TGs × 32 threads.
- `smoke_threadgroup`: threadgroup memory + barrier reduction.

## Section 5 — M1: dense (3.5 min)

### Kernel structure (~1.5 min)

- Mirror `qmv_fast`'s template signature, register-resident
  accumulator, and helper plumbing.
- Adapt for zigzag: per-K-pos "16 weights → 16 row-partials"
  rather than qmv_fast's "16 weights × 16 K-pos → 1 partial."
  `qdot` does not fit; custom inner loop.
- One diagram contrasting the two access patterns.

### Threadgroup atomic_float pivot (~2 min)

- Designed for `threadgroup atomic<float>` per Apple's docs;
  unsupported in MSL.
- Solution: `atomic<int>` + `SCALE_FACTOR = 2^16`, the same
  workaround llama.cpp-SpQt uses (`ggml-metal.metal:5680`).

Result: dense kernel lands within ~10% of `mx.quantized_matmul`
at LLM-relevant shapes.

## Section 6 — M2: sparse + tuning (5.5 min)

### Diff from M1 (~1 min)

- Same threadgroup geometry, within-TG reduce, and atomic output.
  Only the K-walk changes from contiguous to idx-driven.
- Index format: `[n, idx_0, ..., idx_{n-1}]` (count prefixed for
  kernel-side access without a separate `set_bytes`).

### Headline result and sub-linear scaling (~2 min)

- ~1.8× speedup vs `mx.quantized_matmul` at 75% sparsity on
  LLM-FFN shapes; ~1.2× at 50%.
- The theoretical 4× at d=0.25 is not achievable: random-K cache
  miss penalty, plus ~256 fixed cross-TG atomic adds per band per
  launch.
- Linear fit: `sparse(density) ≈ α + β × density × dense`, with
  α ≈ 40% of dense at large shapes.

### Geometry tuning sweep (~2.5 min)

After correctness, a tuning knob remained:
`(num_simdgroups, threadgroups_per_band)`, defaulted to (4, 4).

- Sweep: 11 combos × 7 shapes × 2 densities × 3 runs = 462
  measurements.
- Single-run "best" flipped between (4, 1), (1, 4), (2, 2), and
  (2, 4) per shape — suggesting a shape-dependent dispatch.
- Multi-run: the per-shape "best" was run-to-run noise; (2, 4) is
  universally within 7% of best across all 42 measurements.
- Decision: single rule (2, 4) instead of shape-dependent
  dispatch.

## Section 7 — Results (3 min)

Headline table (reuse from README) covering all 7 shapes with
two regimes:
- ≤2 MB weights: overhead-bound; sparse loses.
- ≥8 MB weights: memory-bound; sparse wins.
- LLM FFN shapes (~22 MB): ~1.8× at 75% sparsity.

## Section 8 — Future work + integration (2 min)

### Deferred from the kernel

- Coarse-tail handling for arbitrary densities — ~30 min of work,
  deferred because the brief asks for a single arbitrary shape.
- Within-SG `simd_sum` cross-lane reduce, multi-batch, CPU
  fallback.
- Quantization-scheme comparison: MLX-affine (gs=64, flat scales)
  and Q4_K (gs=256, hierarchical scales) differ along multiple
  axes; the impact on the SpQt speedup ratio is left for future
  investigation. `group_size=128` in MLX-affine is a natural
  starting point.

### Path to MLX core integration

- Mode-dispatch (`mx.quantized_matmul(mode='affine_zigzag')`)
  breaks for sparse: `idx` has no meaning in the affine mode,
  forcing sparse to be standalone. For API consistency, dense
  should be standalone as well.
- Standalone integration: ~10 hours focused work (Metal-only).
  Primitive into `mlx/backend/metal/quantized_zigzag.cpp`,
  kernels into `mlx/backend/metal/kernels/quantized_zigzag.h`,
  bindings as 4 new functions. Zero regression risk to the
  existing affine path.
- Multi-platform (CPU, CUDA): +6–10 days. PR review cycle: weeks.

## Section 9 — Closing (0.5 min)

- Repository: `github.com/rxpwang/mlx-SpQt`, branch `rxpwang/spqt`.
- Demo: `cd extensions/mlx_spqt && python demo_spqt.py`.
