# `qmv_fast_impl` deep-dive

A line-by-line walkthrough of MLX's `qmv_fast_impl` — the kernel SpQt M1's
dense GEMV mirrors. Written to make the M1 design choices (step 3.3) concrete:
which parts transfer verbatim, which flip for the zigzag layout, and which
adapt with parameter changes.

Source: `mlx/backend/metal/kernels/quantized.h:749-814`. Helpers
(`load_vector`, `qdot`) referenced inline.

## TL;DR

`qmv_fast_impl` is **row-tiled, K-serial** with an "amortize x across rows"
microoptimization:

- **8 output rows per TG** (2 simdgroups × 4 rows).
- Each simdgroup walks K to completion in 512-position chunks.
- Per chunk: load 16 activations once (`load_vector`), then run **4 `qdot`
  calls** (one per row) — same `x_thread` strip reused across all 4 rows.
- Cross-lane reduce per row via `simd_sum`; lane 0 writes 4 outputs.

For SpQt zigzag, **the activation/row roles invert**: instead of "one x strip
× many rows," we'll have "one activation × many row-partials per K-position."
This means **`qdot` itself does not fit our zigzag layout directly** — its
"sum into one accumulator" model can't service `group_size` separate
row-partials (in zigzag, the row-band size = `group_size` by construction).
We'll reuse `load_vector` semantics where applicable, and the bit-extraction
logic from `qdot`'s body, but write our own inner loop.

## 1. Function signature and template parameters

```cpp
template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(
    const device uint32_t* w,                    // packed weights (slot 0)
    const device T*        scales,               // per-group fp scales (slot 1)
    const device T*        biases,               // per-group fp biases (slot 2)
    const device T*        x,                    // activations (slot 3)
    device       T*        y,                    // output (slot 4)
    const constant int&    in_vec_size,          // = K
    const constant int&    out_vec_size,         // = N
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]);
```

**Three template parameters**, all compile-time constants:

| Param | Meaning | Our M2 case |
|---|---|---|
| `T` | activation/output dtype | `half` (fp16) |
| `group_size` | quantization group size | `64` |
| `bits` | quantization bit width | `4` |

The template machinery instantiates this for `(fp32, fp16, bf16) × (32, 64, 128) × (2,3,4,5,6,8) × (batched=0,1)` = ~324 specializations per `affine_qmv_fast_*` kernel binary.

Per Working Principle #4, our M1 zigzag kernel mirrors this template
signature and **adds two tuning knobs**: `<typename T, int group_size,
int bits, int num_simdgroups, int TGs_per_band>`. The row-band size in
zigzag equals `group_size` by construction (the rearrangement makes
`group_size` rows × 1 K-position contiguous, so the standard quantizer
picks up exactly one group per `(band, k_pos)` slice). Mirrors
llama.cpp-SpQt where `superblock_size = QK_K = 256` plays both roles in
Q4_K. No `M_TILE` parameter. The `num_simdgroups` and `TGs_per_band`
parameters are M3-tunable knobs for occupancy and atomic-contention
balance — qmv_fast hardcodes them (`num_simdgroups = 2`,
`results_per_simdgroup = 4`), but for SpQt's two-level reduction we want
them as compile-time constants the host can pick.

## 2. Constexpr derivations

```cpp
constexpr int packs_per_thread       = bits == 2 ? 1 : 2;        // 2 for bits=4
constexpr int num_simdgroups         = 2;                         // hardcoded
constexpr int results_per_simdgroup  = 4;                         // hardcoded
constexpr int pack_factor            = get_pack_factor<bits, 32>();  // 8 for bits=4 (32/4)
constexpr int bytes_per_pack         = get_bytes_per_pack<bits, 32>();// 4 for bits=4
constexpr int values_per_thread      = pack_factor * packs_per_thread;  // 16 for bits=4
constexpr int block_size             = values_per_thread * SIMD_SIZE;   // 512 for bits=4
constexpr int scale_step_per_thread  = group_size / values_per_thread;  // 4 for gs=64, vpt=16
```

For `bits=4, gs=64`:

| Constant | Value | Meaning |
|---|---|---|
| `packs_per_thread` | 2 | each thread holds 2 uint32-equivalent packs of weights |
| `num_simdgroups` | 2 | 2 simdgroups per threadgroup |
| `results_per_simdgroup` | 4 | each simdgroup produces 4 output rows |
| `pack_factor` | 8 | 8 nibbles per uint32 |
| `bytes_per_pack` | 4 | 1 uint32 = 4 bytes |
| `values_per_thread` | 16 | each thread processes 16 weights per K-block iteration |
| `block_size` | 512 | total K-positions covered per outer-loop iteration (16 × 32 lanes) |
| `scale_step_per_thread` | 4 | 4 consecutive lanes share one scale (since 16 weights/thread × 4 = 64 = group_size) |

**These derivations transfer verbatim to our zigzag kernel.** Same formulas; `pack_factor`, `bytes_per_pack`, `values_per_thread`, etc. are pure functions of `bits` and `group_size`. `block_size` and `scale_step_per_thread` may need reinterpretation under zigzag's flipped semantics, but the math is identical.

## 3. Thread positioning

### Output row assignment

```cpp
const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                    simd_gid * results_per_simdgroup;
//                      = tid.y * 8 + simd_gid * 4    for bits=4
```

For threadgroup at `tid.y`:
- Simdgroup 0 (`simd_gid = 0`) handles output rows `[8*tid.y .. 8*tid.y + 3]`
- Simdgroup 1 (`simd_gid = 1`) handles output rows `[8*tid.y + 4 .. 8*tid.y + 7]`

Each simdgroup owns 4 contiguous output rows. The 32 lanes within a simdgroup all work on the same 4 rows in parallel.

**Grid dispatch** (from `quantized.cpp:251-254`):
```cpp
int bn = 8;       // out rows per TG
int bk = 32;      // lanes per simdgroup (= SIMD_SIZE)
MTL::Size group_dims(bk, 2, 1);                  // 32 lanes × 2 SG = 64 threads/TG
MTL::Size grid_dims(M, ceil(N/bn), B);           // M=batch, N/8 = num row-tiles, B=batch leading dims
```

For our shape (M=1, K=4096, N=4096): grid is `(1, 512, 1)` — 512 TGs along the row-tile axis, each handling 8 output rows.

### Pointer setup

```cpp
const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;  // K * 4 / 8 = K/2 bytes per row
const int in_vec_size_g = in_vec_size / group_size;                     // K / 64 groups per row

ws     += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
x      += tid.x   * in_vec_size   + simd_lid * values_per_thread;
y      += tid.x   * out_vec_size  + out_row;
```

Each pointer advances to "this thread's starting position within its simdgroup's 4 rows":

- **`ws`**: row stride = `K/2` bytes. Lane offset = `simd_lid * 8` bytes (each lane handles 16 nibbles = 8 bytes per row per K-iter).
- **`scales`**, **`biases`**: row stride = `K/64`. Lane offset = `simd_lid / 4` (4 lanes share each scale because 16 weights/thread × 4 lanes = 64 = group_size).
- **`x`**: lane offset = `simd_lid * 16` (each lane reads 16 contiguous activations).
- **`y`**: just the row offset (output write target).

The interleaving (4 lanes/scale, 32 lanes/512 K-positions, 32 lanes for bytes) is what makes 16 weights/thread the natural unit.

## 4. The "load x once, FMA into 4 rows" pattern (the heart)

```cpp
for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);  // ★ ONCE per outer iter

    for (int row = 0; row < results_per_simdgroup; row++) {            // 4 rows
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);  // ★ 4 calls, same x_thread
    }

    ws     += block_size * bytes_per_pack / pack_factor;   // step weights forward
    scales += block_size / group_size;
    biases += block_size / group_size;
    x      += block_size;
}
```

This is the core microoptimization:

- **`load_vector` runs ONCE per outer iter**, populating `x_thread[16]` with the same 16 pre-scaled activations.
- **The inner loop runs 4 `qdot` calls**, each using a different row's weights/scale/bias but the **same `x_thread` strip**.
- Activation bandwidth is amortized 4× across the 4 output rows.

Without this pattern (e.g., 1 row per simdgroup): each row's `qdot` would re-load the same 16 activations from device memory. With it: load once per simdgroup, reuse 4×.

The constexprs `num_simdgroups=2, results_per_simdgroup=4` were tuned exactly for this: 2 SG × 4 rows = 8 rows/TG, all sharing K-bandwidth amortization.

### Pointer advancement per outer iter

```cpp
ws     += block_size * bytes_per_pack / pack_factor;   // 512 * 4 / 8 = 256 bytes
scales += block_size / group_size;                      // 512 / 64 = 8 groups
biases += block_size / group_size;
x      += block_size;                                   // 512 floats
```

Each iter advances by one `block_size`-worth of K. After `K / block_size` iters (= 8 iters for K=4096), the simdgroup has covered the full K dimension for its 4 rows.

## 5. The `qdot` inner call

`qdot<U, values_per_thread=16, bits=4>` (`quantized.h:191-290`) computes for one row:

```
result += sum_i (x_thread[i] * w_i_dequant) for i in 0..15

where w_i_dequant interpretation: scale * (q[i]) + bias
                                  with scale and bias passed as scalars (per-group)

Concretely returns: scale * accum + sum * bias
              where accum = sum_i (x_thread[i] * q[i]) [bit-extracted]
                    sum   = sum_i (x[i])               [original, pre-scale]
```

For `bits=4`, `qdot` reads `ws[i] = uint16_t` (2 nibbles' worth) and AND-masks
nibbles in their natural bit positions:

```cpp
// for i in 0..3:
accum += x_thread[4*i  ] * (ws[i] & 0x000f) +     // nibble at bits  0..3
         x_thread[4*i+1] * (ws[i] & 0x00f0) +     // nibble at bits  4..7
         x_thread[4*i+2] * (ws[i] & 0x0f00) +     // nibble at bits  8..11
         x_thread[4*i+3] * (ws[i] & 0xf000);      // nibble at bits 12..15
```

The bit-position trick is paired with `load_vector`'s pre-scaling of `x_thread`:

```cpp
// load_vector for bits=4:
x_thread[i+0] = x[i+0];
x_thread[i+1] = x[i+1] / 16.0f;     // compensates for & 0x00f0 = nibble * 16
x_thread[i+2] = x[i+2] / 256.0f;    // compensates for & 0x0f00 = nibble * 256
x_thread[i+3] = x[i+3] / 4096.0f;   // compensates for & 0xf000 = nibble * 4096
```

Net effect: `x_thread[i+1] * (ws[k] & 0x00f0) = (x[i+1]/16) * (nibble * 16) = x[i+1] * nibble`. The bit-position factors cancel, math is correct.

**`qdot` is hardwired for "16 weights = 16 K-positions of ONE row"** — this is the assumption that breaks under zigzag.

## 6. Output write

```cpp
for (int row = 0; row < results_per_simdgroup; row++) {  // 4 rows
    result[row] = simd_sum(result[row]);                  // 32-way reduce within SG
    if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);             // lane 0 of each SG writes
    }
}
```

After the K-loop completes, each lane has accumulated 4 partial sums (one per row). `simd_sum` reduces across all 32 lanes of the simdgroup. Lane 0 writes the final 4 outputs to `y[out_row + row]`.

**No cross-TG or cross-simdgroup synchronization needed** — each (TG × simdgroup × row) pair fully owns its 4 output rows. This is what makes qmv_fast simple: no atomics, no shared memory, no barriers.

## 7. Mirror / flip / adapt analysis vs. SpQt zigzag

Per Working Principle #4, identifying which parts of `qmv_fast_impl` transfer
to our M1 zigzag-GEMV kernel:

| Element | qmv_fast | SpQt zigzag (M1) | Verdict |
|---|---|---|---|
| Template signature | `<T, gs, bits>` | `<T, gs, bits, num_simdgroups, TGs_per_band>` | **Adapt** — add tuning knobs |
| Constexpr derivations (`pack_factor`, `bytes_per_pack`, `values_per_thread`, `scale_step_per_thread`) | as shown | unchanged formulas | **Mirror verbatim** |
| `block_size` constant | 512 (= 16 × 32 lanes) | reinterpreted: how many K-positions one SG iteration covers | **Adapt** — same idea, semantics shift |
| `typedef float U; thread U result[N]` accumulator | 4 partials per thread | `rows_per_lane = group_size / SIMD_SIZE` partials per thread (= 2 for gs=64) | **Mirror, dimension changes** |
| `load_vector<T, U, N, bits>` | loads 16 x values, pre-scales for bit-position trick | M2-relevant: loads x at sparse K-indices; M1 dense: scalar `x[k]` per K-position is sufficient | **Optional reuse** in M1; M2 may revisit |
| `qdot<U, N, bits>` | 16 weights × 16 K-pos × 1 row → 1 partial | 1 uint32 × 1 K-pos × `rows_per_lane` rows → `rows_per_lane` partials | **❌ Does NOT fit.** See §8. |
| Bit-extraction logic (mask constants `0x000f`, `0x00f0`, etc.) | inside `qdot` | reused outside qdot in our custom inner loop | **Reuse fragments** |
| K-walk: `for k = 0; k < in_vec_size; k += block_size` (contiguous) | one SG walks all K | M1 dense: SG walks its K-stripe `[k_start, k_end)`; M2: idx-driven | **Mirror shape; semantics shift to per-SG K-stripe** |
| Output write (`simd_sum` per row, lane 0 writes) | direct `y[row] = result[row]` | within-TG reduce (TG-mem + barrier) → SG 0 atomic-add to `y` | **❌ Replace** with two-level reduce |
| Threadgroup geometry (2 SG × 4 rows = 8 rows/TG) | 2 simdgroups, 4 rows/SG, single TG owns the rows | **4 simdgroups × 32 lanes/SG = 128 threads/TG**; all SGs cover same `group_size` rows of one band, splitting K | **Adapt** — see §9 |
| Activation reuse: load x once, FMA into 4 rows | 4× amortization across 4 rows | one activation per K-position broadcast across `rows_per_lane` rows per lane (and across all lanes of the SG) | **Mirror, dimension flips** |
| Within-TG synchronization | none | required (per-row partials summed across simdgroups via threadgroup memory + barrier) | **❌ New requirement** (M0b smoke #4 verified) |
| Cross-TG synchronization | none (rows fully owned by simdgroup) | required (multiple TGs contribute to same output rows via K-tiling) | **❌ New requirement** (M0b smoke #3 verified) |

## 8. The qdot-fits question — answered

**`qdot` does NOT fit our zigzag layout directly.**

The fundamental incompatibility:

- **`qdot`'s assumption**: 16 weights consumed per call all belong to **one
  output row** at 16 different K-positions. Output: one accumulator (one row's
  contribution to its dot product).
- **Zigzag layout's structure**: 16 weights at one memory position correspond
  to **16 different output rows** at one K-position. Output: 16 separate
  accumulators (one per row), all with different scales/biases.

Even reading 16 nibbles from a uint32-pair, qdot's math (`scale * accum + sum * bias`) lumps them into a single scalar result, weighted by ONE scale and ONE bias. Zigzag has 16 different scales and 16 different biases — one per row.

You can't reuse `qdot`'s wrapper. You have to write your own inner loop that:

1. Reads `group_size` nibbles for one K-position (a contiguous run in zigzag layout).
2. For each row in the band, dequantizes (`scale * nibble + bias` — a *single* shared scale and bias for this `(band, k_pos)` slice) and accumulates `x[k] * dequant` into that row's partial.

What you *can* reuse:

- **`load_vector`** for loading and pre-scaling x values *if* a K-block of multiple K-positions is being processed at once (M2's idx-driven walk may use this).
- **The bit-extraction mask logic** from `qdot`'s body (`& 0x000f`, `& 0x00f0`, etc.) — this is the format-specific unpacking that's worth not re-deriving for `bits ∈ {3, 5, 6}`.
- **The pre-scaling-trick semantics** — if our inner loop uses pre-scaled `x_thread` values, we can use the same AND-mask-without-shift pattern.

**Implication for Working Principle #4**: we lose the "use qdot directly" reuse. But the deeper structural conventions (template signature, accumulator types, simdgroup reduction patterns, helper function shapes) are all preserved. Working Principle #4 still applies; just at a slightly lower level than calling `qdot` outright.

## 9. Threadgroup geometry recommendation for M1

`qmv_fast_impl`'s geometry: 2 SG × 4 rows = 8 rows/TG, single TG per row-tile (no cross-TG sync).

Our zigzag M1 needs to handle:
- **64 rows per band** (= `group_size`, by construction).
- **Multiple SGs per TG covering the same band, splitting K** — mirrors
  llama.cpp-SpQt's reference and reduces atomic contention vs. the
  qmv_fast-style "1 lane = 1 row" alternative.
- **K-tiled across TGs** for the cross-TG atomic-reduce architecture (per
  llama.cpp-SpQt's reference and §3 of `mlx-spqt-reference.md`).
- **Within-TG reduction** to sum per-row partials across the simdgroups in a TG.

Recommended starting geometry for M1:

| Quantity | Value | Reasoning |
|---|---|---|
| simdgroups per TG | 4 | matches SpQt reference's per-TG SG count; M3 may tune in 2-8 range |
| lanes per SG | 32 | hardware (`SIMD_SIZE`) |
| threads per TG | 128 | `num_simdgroups × SIMD_SIZE` |
| rows per lane | `group_size / SIMD_SIZE` = **2** | each lane owns 2 contiguous rows of the band |
| rows per simdgroup | `group_size` = **64** | each SG produces partials for **all** rows of the band (over its K-stripe) |
| TGs per (output band × full K-walk) | **4** | starting recommendation; 2-8 tunable in M3 |
| K-positions per SG iteration | `K / (num_simdgroups × TGs_per_band)` | = 4096 / 16 = 256 per SG |
| K-positions per TG iteration | `K / TGs_per_band` | = 4096 / 4 = 1024 per TG |

For our shape (M=4096, K=4096, group_size=64):
- Total bands = M / `group_size` = 64
- TGs per band = 4 (recommended start; 2-8 is the tunable range)
- Total TGs = 64 × 4 = 256
- Total threads = 256 × 128 = **32768** (same total work as alternative geometries; just packed differently)
- Threadgroup memory per TG = `num_simdgroups × group_size × 4 B` = **1 KB** (per-SG row-partial regions for the within-TG reduce)

Final tuning will happen in M3; M1's correctness target works for any
reasonable `(num_simdgroups, TGs_per_band)` combination.

## 10. Open questions answered

From `mlx-spqt-reference.md` §4:

1. **Does qdot fit zigzag layout?** ❌ No — qdot's "16 weights → 1 accumulator" doesn't match zigzag's "16 weights → 16 accumulators" pattern. Custom inner loop needed.
2. **qmv_fast-style or SpQt-reference-style threadgroup pattern?** **SpQt-reference-style** chosen: multiple SGs per TG split a band's K-walk, within-TG reduce sums per-row partials across SGs. Higher per-TG occupancy, lower atomic contention (4 TGs/band vs. 8), better data locality, and we already paid the cost of building the within-TG-reduce smoke kernel (M0b smoke #4) so the pattern is verified.
3. **Right threadgroup geometry for `group_size`=64?** Starting recommendation: 4 SG × 32 lanes = 128 threads/TG (each lane owns 2 rows of the band); 4 TGs per band. Tunable in M3 (`num_simdgroups` range 2-8; `TGs_per_band` range 2-8).
4. **Cross-TG `atomic_float`?** ✅ Yes (already verified by M0b smoke #3). Use directly; skip the int-scaling workaround.
5. **Within-TG reduce mechanism?** ✅ `threadgroup atomic<float>` + two `threadgroup_barrier`s (Option I — see `M1-zigzag-layout.md` §0). Native MSL atomic on Apple GPU family 7+ (M1+); MLX core's `atomic.h` treats `float` as a first-class atomic type. All SGs atomic-add their per-row partials into a 64-slot shared TG-mem accumulator; SG 0 atomic-loads the totals and atomic-adds them to the global output. Cleaner than per-SG disjoint regions + manual sum (Option II) at this row-tile size, with no measurable contention since 64 slots × 4 SGs × 32 lanes is small.
6. **Single-scale-per-group broadcasting?** In zigzag, *one* fp16 scale + *one* fp16 bias is shared across all `group_size` rows of a band at a given K-position (since the row-band size = `group_size` by construction). Each SG loads 1 scale + 1 bias per K-position, broadcast across all 32 lanes — naturally cached after the first lane reads.

## 11. M1 design implications

Synthesis ready for step 3.3 (the M1 design doc):

- **Layout**: zigzag block = `group_size × group_size` = 64 × 64 weights (the row-band size = `group_size` by construction), packed M-major within (64 rows' nibbles for one K-position consecutive in memory). Plus *one* shared scale and *one* shared bias per `(band, K-position)` slice.
- **Kernel template**: `zigzag_qmv_dense<T, gs, bits, num_simdgroups, TGs_per_band>` — extends `qmv_fast_impl`'s `<T, gs, bits>` with two tuning knobs. Instantiate `<half, 64, 4, 4, 4>` for our M1 shape.
- **Threadgroup**: 4 SG × 32 lanes = 128 threads/TG; each lane owns 2 rows; all SGs cover all 64 rows, splitting K. 4 TGs per band.
- **Inner loop**: custom (not qdot). Per K-position, each lane reads 1 uint32 (containing `rows_per_lane` = 2 nibbles for its 2 rows); dequant with the shared `(scale, bias)` and FMA into per-lane row-partials.
- **Two-level reduce**: within-TG via threadgroup memory + barrier (smoke #4); cross-TG via `atomic_fetch_add_explicit` on `device atomic_float*` (smoke #3).
- **Threadgroup**: 4 SG × 32 lanes = 128 threads/TG; `rows_per_lane = group_size / SIMD_SIZE = 2`. All 4 SGs cover same 64 rows of one band, splitting K. 4 TGs per band.
- **Cross-TG reduce**: `atomic_fetch_add_explicit` on `device atomic_float* out`. M0b smoke #3 confirms this scales lossless.
- **Pre-zero output**: explicit zero-fill in `eval_gpu` before main dispatch (per Phase 6 recommendation).
- **What we mirror from qmv_fast**: template signature shape, constexpr derivations, accumulator typedef pattern, simdgroup reduction conventions, pointer-advancement style.
- **What we replace**: qdot call → custom dequant+FMA loop; direct output write → atomic-add cross-TG.

Step 3.3 (the M1 design doc) integrates this with the SpQt reference layout to produce the final spec for kernel + zigzag_quantize.
