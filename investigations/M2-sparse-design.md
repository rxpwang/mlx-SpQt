# M2 — Sparse zigzag-GEMV: idx-driven K-walk

The sparse counterpart to M1's dense zigzag-GEMV. Same layout, same
threadgroup geometry, same within-TG and cross-TG reduction — only the
K-walk mechanism changes from contiguous to idx-driven. This is the
milestone where SpQt's "skip K-positions to save bandwidth" idea
materializes.

After review/sign-off, this doc is the contract M2's implementation
(commit `93e86646`) builds against.

## Goal

Produce a registered MLX custom op `mlx_spqt.zigzag_qmv_sparse(x, idx,
w_zz, scales, biases)` that computes the same logical math as
`mlx_spqt.zigzag_qmv_dense` but only walks K-positions in `idx`. Skipped
K-positions cost no kernel time, no memory bandwidth, no atomic ops.

**Done criterion**: `(y_sparse - y_ref).abs().max() < 1e-3`, where
`y_ref = x[:, idx] @ dequantize(w_zz)[:, idx].T` (fp32 throughout).

## 1. Index format

`sparse_indices`: `int32` array of shape `(n + 1,)`:

| Index | Content |
|---|---|
| `0` | `n` — count of active K-positions |
| `1..n+1` | sorted K-positions in `[0, K)` |

The +1 prefix lets the kernel read `n_total` from `sparse_indices[0]`
without needing a separate `set_bytes(idx_count, ...)` call. Mirrors
llama.cpp-SpQt's reference convention (`ggml-metal.metal`).

Constraint (current MVP): `n % 16 == 0`. Densities like 0.25, 0.5,
0.75, 1.0 satisfy this for all our K values (multiples of 256). Mixed
densities (0.6, 0.8, etc.) need the deferred coarse-tail handling
(§5).

## 2. Op signature

```python
mlx_spqt.zigzag_qmv_sparse(
    x,                # (B, K) fp16
    sparse_indices,   # (n+1,) int32
    w_zz,             # (M/gs, K, gs * bits / 32) uint32 — full layout (not subsetted)
    scales,           # (M/gs, K) fp16
    biases,           # (M/gs, K) fp16
    group_size=64,
    bits=4,
) -> (B, M) fp32
```

Note that `w_zz`, `scales`, `biases` are the **full** zigzag layout
covering all K-positions. The kernel reads only the subset specified
by `idx`. No precomputation needed when `idx` changes — same weights
serve all idx selections.

## 3. Kernel structure (deltas from M1)

What's **identical** to M1 dense:
- Threadgroup geometry: 4 SG × 32 lanes = 128 threads/TG, 4 TGs/band.
- `packs_per_thread = 4`, `values_per_thread = 32` rows/lane/iter.
- Lane → row-chunk mapping: 16 × 2 (K-pos × row-chunk) grid per SG.
- Within-TG reduce: `threadgroup atomic<int>` + SCALE_FACTOR=2^16.
- Cross-TG reduce: `device atomic<float>` for global output.
- Bias hoist (`x_bias_cur` accumulated into `bias_total`, applied once at end).

What **changes** from M1 dense:

### 3.1 K-walk: idx-driven instead of contiguous

```metal
// M1 dense:
const int k_per_sg = K / (TG × NSG);
const int k_start  = ...;                            // contiguous K-stripe
for (int k = k_start; k < k_end; k++) {
    // ... use k as the K-position ...
}

// M2 sparse:
const int n_total           = sparse_indices[0];
const int groups_per_sg     = n_total / (TG × NSG);
const int groups_start_id   = groups_per_sg × sg_global_id;
int group_cur_count_thread  = groups_start_id + simd_lid / 2;  // lane's slot in idx[]

for (int outer = 0; outer < total_step; outer++) {
    if (group_cur_count_thread < groups_start_id + groups_per_sg) {
        // Read the actual K-position from the index array
        int k = sparse_indices_cur[group_cur_count_thread];

        // ... use k normally to index x, scales, biases, w_zz ...
    }
    group_cur_count_thread += 16;
}
```

Each lane partitions the **idx array** (not K), reads `idx[i]` to get
its K-position, then indexes everything else at that K-position.
Adjacent lanes within an SG cover adjacent idx entries; lanes
overshoot per outer iter by `values_per_thread × 32 / group_size = 16`
positions in idx-space.

### 3.2 Per-SG bounds check (handles fine tail)

For some `n_total` values, `groups_per_sg` doesn't divide evenly by 16
(the per-iter SG step). Without a bounds check, lanes in the last
outer iter overshoot into the **next SG's range** — same idx position
gets processed twice (by the current SG AND its neighbor) → double
counting → wrong results.

**Fix**: each lane's bounds check is against `groups_start_id +
groups_per_sg`, not `n_total`. Lanes whose `group_cur_count_thread`
exceeds the SG's own end-of-range simply skip the iteration; their
acc/bias contributions stay at 0 → no contribution to the band's sum.

This was the bug-of-the-week during M2 implementation — see §6.B
below for the full story.

### 3.3 Per-iter pointer math (more expensive than M1)

In M1 dense, pointers advance by a fixed stride per iter
(`x += stride`, `ws += bytes_per_K_pos`, etc.) — cheap.

In M2 sparse, pointers must be **recomputed per iter** from the
new K-position:
```metal
// Per-iter, after reading k = sparse_indices_cur[group_cur_count_thread]:
float x_cur     = x[k];
int   scl_off   = k + K * band_id;
float x_scale   = x_cur * scales_zz[scl_off];
float x_bias    = x_cur * biases_zz[scl_off];
const device uint16_t* w_cur = (const device uint16_t*)(
    ws + row_start * K * bits / 8
       + k * group_size * bits / 8
       + ((simd_lid * vpt % gs) * bits / 8)
);
```

Each per-iter requires one multiplication for the band offset, one
multiplication for the K-pos offset, and one addition for the lane's
row-chunk offset. About 3-4 extra ops per inner iteration vs M1's
pointer-advance pattern. Compounds with cache miss penalty for
non-contiguous K-walking.

## 4. Bench results

Multi-shape × multi-density sweep. Per-kernel time (μs) at LOOPS=32 with
`(num_simdgroups, threadgroups_per_band) = (2, 4)` (the default after the
tuning sweep in §5; matches the M1 dense kernel's (4, 4) within ~5% at
LLM shapes):

| Shape | weights | dense | mlx_qmv | s@1.0 | s@0.75 | s@0.50 | s@0.25 |
|---|---|---|---|---|---|---|---|
| (1024, 1024) | 0.5 MB | 15 | 12 | 15 | 15 | 15 | 15 |
| (4096, 4096) | 8 MB | 48 | 34 | 50 | 41 | 36 | 32 |
| (8192, 8192) | 32 MB | 174 | 164 | 178 | 169 | 139 | **98** |
| (4096, 11008) | 22 MB | 120 | 111 | 122 | 115 | 99 | **63** |
| (11008, 4096) | 22 MB | 121 | 112 | 123 | 118 | 98 | **69** |
| (4096, 16384) | 32 MB | 179 | 167 | 180 | 166 | 139 | **97** |

### 4.1 Speedup vs `mx.quantized_matmul` at production sparsity

| Shape | sparse@0.25 | mlx_qmv (full) | speedup |
|---|---|---|---|
| (8192, 8192) | 98 μs | 164 μs | **1.67×** |
| (4096, 11008) | 63 μs | 111 μs | **1.76×** |
| (11008, 4096) | 69 μs | 112 μs | **1.62×** |
| (4096, 16384) | 97 μs | 167 μs | **1.72×** |

**Average ~1.7× speedup at 75% sparsity** on LLM-relevant FFN shapes.
This is the SpQt headline win: skipping 75% of K-positions at runtime
by walking only the active subset.

### 4.2 Sub-linear scaling with density

If the kernel were purely memory-bound on weight bandwidth, sparse
time would scale linearly with density (`sparse(d) ≈ d × dense`). It
doesn't quite:

| Density | Theoretical | M=K=8192 observed | Achieved fraction |
|---|---|---|---|
| 1.0 | 178 μs | 178 μs | 100% |
| 0.5 | 89 μs | 139 μs | 64% |
| 0.25 | 45 μs | 98 μs | 46% |

At density 0.25 we achieve ~46% of theoretical, not 25%. Two reasons:

**(a) Random-K cache miss penalty.** Sparse K-positions are not
contiguous in memory; consecutive idx entries access different cache
lines in `x`, `scales`, `biases`, and `w_zz`. L1 hit rate degrades.
Inherent to sparse access patterns — same penalty seen in
llama.cpp-SpQt's measurements.

**(b) Fixed per-TG overhead.** The kernel does ~256 cross-TG
atomic-float adds per band per launch (4 TGs × 64 row slots), plus
`shared_out` init and zero-fill of the output buffer. These costs
don't scale with density. Linear-fit for M=K=8192:
`sparse(density) ≈ α + β × density × dense`, with α ≈ 2.27 ms (40%
of dense). At density=0.25 the fixed cost is the larger fraction of
total time.

This is the honest "what we ship" result. SpQt's value is real (~1.7×
at 75% sparsity), but the theoretical 4× speedup at 25% density is
unachievable at this kernel design without further optimization
(consolidate atomics, larger per-TG K-stripes, etc.).

## 5. Threadgroup geometry tuning — methodology and default selection

### 5.1 Motivation

The kernel template is parameterized on `(num_simdgroups,
threadgroups_per_band)`. The .metal file pre-instantiates 11 combos:
`{(2,2), (2,4), (4,1), (4,2), (4,4), (4,8), (8,2), (1,4), (2,1),
(2,8), (8,1)}`. The Python op exposes `num_simdgroups` and
`threadgroups_per_band` as runtime kwargs (default `(2, 4)`),
selected via `mx::concatenate`-built kernel name.

Sweep tooling: `extensions/mlx_spqt/sweep_sparse_geometry.py` runs
all 11 combos × 7 shapes × 2 densities (`{0.5, 0.25}`), checks
correctness and times each cell. Total: 154 measurements per run.

### 5.2 Sweep results (3 independent runs)

Per-kernel μs at LOOPS=32, density=0.25 column shown (density=0.5 is
similar). **Best per (shape, run)** in bold:

| Shape | (2,2) | (2,4) | (4,1) | (4,2) | (4,4) | (4,8) | (8,2) | (1,4) | (2,1) | (2,8) | (8,1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (1024, 1024) | 12 | 12-13 | **12** | 12-13 | 15 | 19 | 15 | **12** | 12-13 | 14-16 | 12-13 |
| (2048, 2048) | 15 | 15-22 | 14-17 | 16 | 20-22 | 28-29 | 22-24 | **14-15** | 16 | 19 | 16 |
| (4096, 4096) | **24** | 25-26 | 25-26 | 25-26 | 32 | 49-50 | 33-34 | **24-26** | 29-30 | 32-33 | 28-29 |
| (8192, 8192) | 97-101 | 99-103 | 106-109 | 100-101 | **95-97** | 97-100 | 96-101 | 97-103 | 99-103 | 96-97 | 101-105 |
| (4096, 11008) | 59-68 | **59-67** | 59-71 | 65-71 | 60-65 | 61-66 | 64-67 | **59-63** | 78-84 | 64-66 | 73-75 |
| (11008, 4096) | 63-64 | 62-63 | 62-65 | **62-63** | 69-71 | 111 | 71-72 | 63 | **60-61** | 70 | 62-63 |
| (4096, 16384) | 96-99 | **97-98** | **97** | 105-106 | 97-98 | 95-99 | 102-107 | 96-100 | 124-129 | 99-103 | 111-118 |

The "best" combo flips between runs at most cells — consecutive runs
pick different winners that are within run-to-run noise of each other
(typically ±2 μs). This is **expected at this resolution** and
suggests we shouldn't over-fit a per-shape lookup.

### 5.3 (2, 4) is universally near-optimal

Across all 14 (shape, density) cells × 3 runs (= 42 measurements),
(2, 4)'s gap from the per-cell best:

| Worst gap | Median gap | Cells where (2, 4) tied for best |
|---|---|---|
| **+7%** | **0%** (tied) | 17 of 42 |

(2, 4) is never more than 7% off the per-run best in any cell, and is
the actual winner in nearly half the cells. **No other combo is this
robust across the shape range.**

### 5.4 Pattern: NSG × TG ≈ 8 hits the sweet spot

Grouping the combos by `NSG × TG` (= per-band TG count):

| NSG × TG | Combos | Win frequency |
|---|---|---|
| 4 | (4,1), (2,2), (1,4) | Tiny shapes (≤2048²) |
| **8** | **(2,4), (4,2), (8,1)** | **Medium + large (≥4096²) — most cells** |
| 16 | (2,8), (4,4) | Occasional, rarely dominant |
| 32 | (4,8) | Rare wins, fails on (11008, 4096) |
| 2 | (2,1) | Niche — wins (11008, 4096) once |

The intuition:
- **Below 4 per-band TGs**: not enough TG-level parallelism; large
  shapes underutilize compute units.
- **Above 16 per-band TGs**: cross-TG atomic-add fanout dominates;
  each band's output gets too many atomic-float adds.
- **NSG × TG ∈ {4, 8}**: balances both. (2, 4) sits at the upper end
  of this sweet spot and captures most of the wins.

Within `NSG × TG = 8`, **NSG = 2** is preferred over NSG = 4 or 8 at
larger shapes — moderate per-TG thread count (64) gives better GPU
latency hiding than 128 or 256.

### 5.5 Default decision

**Default `(num_simdgroups, threadgroups_per_band) = (2, 4)`.**

Reasoning:
- Within 7% of per-cell best in 100% of measurements.
- Single rule, no shape-dependent dispatch logic in the C++ side.
- Matches the "NSG × TG = 8, NSG = 2" sweet-spot pattern.
- Power users can override via kwargs if benching a specific shape.

Decision against shape-dependent dispatch:
- The 5-7% gap between (2, 4) and the per-cell best is below run-to-run
  noise.
- Adding shape-dependent logic in C++ creates maintenance burden for
  a 5% gain at niche shapes.
- The deliverable's value-prop is the SpQt mechanism, not perfect
  microoptimization.

## 6. What's deferred for the MVP

### Coarse tail (n_total % 16 ≠ 0)

When the total idx count doesn't divide evenly by `total_sgs = 16`,
the current `groups_per_simdgroup = n_total / 16` truncates and **idx
positions at the end never get assigned to any SG**. Affects densities
like 0.6, 0.7, 0.8.

The fix is block-cyclic distribution of the `n_total % 16` remainder
across the first few SGs:
```metal
const int base   = n_total / total_sgs;
const int extra  = n_total - base * total_sgs;
const int my_count = base + (sg_global_id < extra ? 1 : 0);
const int my_start = base * sg_global_id + min(sg_global_id, extra);
```
Combined with the per-SG bounds check (already in place for the fine
tail), this would handle arbitrary `n_total` cleanly.

Estimated effort: ~30-45 minutes of kernel work + verification.
Deferred for the MVP since only single-shape correctness is required.

### Other deferred optimizations

- **Within-SG `simd_sum` cross-lane reduce**: across SGs, lanes 0/2/4/...
  contribute partials for the same row chunk via separate atomic_int
  adds. Coalescing within an SG before the threadgroup atomic could
  reduce TG-atomic ops by ~8× per slot. Likely modest speedup at
  memory-bound shapes.
- **`load_vector`-style activation amortization**: load multiple x's
  per outer iter to amortize idx-driven address computation cost.
  Probably noticeable for sparse where the per-iter computational
  overhead is higher than dense.
- **Multi-batch (B>1)**: kernel locked to B=1; would need batch
  dimension in dispatch grid + idx layout (one idx per batch).
- **Cache-aware idx ordering**: sort/bucket idx for better cache hit
  rate. Limited gain since zigzag layout already groups (band, k_pos)
  contiguously per K-position.

## 7. Implementation experience — lessons captured

### A. The per-SG bounds bug (sparse-specific 2D-aware indexing)

Initial bound check was `if (group_cur_count_thread < n_total)`. This
caught the case where lanes overshoot past the **end of `sparse_indices`**
(global tail), but missed the case where lanes overshoot **into the
next SG's range** (per-SG fine tail).

The bug manifested as: at K=11008, density=0.5 (n_total=5504, divisible
by 16 → coarse-clean, but 5504/16=344 NOT divisible by 16 → fine-tail
present), SG 0's last iter overshot by 8 positions into SG 1's
territory. Both SGs processed those 8 idx positions → 2× double counting
→ test err = 39.5 (vs typical ~5e-4).

Fix: bound by `groups_start_id + groups_per_simdgroup` (the SG's own
end), not `n_total`. Each idx position now belongs to exactly one SG.

Lesson generalizes from M1: **whenever lanes are tiled into a 2D grid
(SGs × within-SG positions), bounds checks need to reflect both
levels**. M1 had a 2D row × K-pos grid; M2 has a similar issue for
SG range × within-SG iter count. Same divmod-decomposition reasoning.

### B. Index format choice — embedded vs out-of-band count

We chose `sparse_indices = [n, idx_0, idx_1, ..., idx_{n-1}]` (n+1
length, count embedded) over the alternative `sparse_indices = [idx_0,
..., idx_{n-1}]` + separate `set_bytes(n)` for the count.

Why: embedded count keeps the kernel signature minimal (one fewer
buffer slot), matches llama.cpp-SpQt's pattern, and the +1 element
overhead is negligible. Trade-off: caller must remember to prepend
the count when constructing the array.

### C. Sub-linear density scaling is fundamental, not a bug

The first time the bench showed sparse@0.25 at ~50% of dense (not
25%), my instinct was "we have a bug — sparse should scale
linearly." Turns out the cache miss penalty + fixed per-TG cost are
the right explanation, and tuning (NSG, TG) didn't help. Lesson:
**before chasing optimization, model the kernel's expected
performance**. If the model predicts the gap, the gap is real.

The "α + β × density × dense" linear fit was a useful diagnostic —
isolating the constant overhead from the variable cost made the
diminishing-returns picture obvious.

### D. Multi-run measurement is essential at sub-10% resolution

The first single-run sweep showed (4, 4) winning at one shape, (2, 4)
at another, (4, 2) at a third — looking like a clear case for shape-
specific defaults. Across 3 independent runs, the "best" combo at
most shapes flipped between equivalent options within run-to-run
noise. The actual signal (NSG × TG ≈ 8 is the sweet spot, with NSG=2
preferred) only emerged after the multi-run aggregation.

Lesson: **single-run GPU bench data at < 10% resolution is unreliable.**
Apple Silicon's clock state, thermal state, and L1/L2 cache state
shift between invocations enough to make sub-10% differences look
significant when they're noise. The §5.3 "(2, 4) within 7% of best"
analysis is robust precisely because it's computed across 42
measurements (14 cells × 3 runs), not 14.

The corollary: **don't add shape-dependent dispatch logic for
differences that don't survive multi-run aggregation.** The complexity
isn't worth the noise-floor "gain."

## Summary

M2 is a small structural diff from M1 (only the K-walk changes), but
delivers SpQt's headline value: **~1.7× speedup over `mx.quantized_matmul`
at 75% sparsity** on LLM FFN shapes. Sub-linear density scaling is
inherent to memory-bound sparse kernels and is well-characterized.

The op is registered as an MLX Custom Operation via the same
extension pattern as M1; correctness is verified across 7 shapes ×
2 densities (14 total cases); bench data covers 7 shapes × 4
densities (28 datapoints). Threadgroup geometry was swept across 11
combos × 7 shapes × 3 runs (462 measurements); the default
`(num_simdgroups, threadgroups_per_band) = (2, 4)` is universally
near-optimal (within 7% of per-cell best). Tail handling for
arbitrary densities and within-SG reduction optimization are deferred
to post-MVP.
