# SpQt reference (llama.cpp-SpQt) — what zigzag concretely means

A focused dive into the llama.cpp-SpQt reference implementation to extract the
*concrete* meaning of "zigzag layout" and how the dense GEMV kernel walks it.
Step 1 of the M1 design sequence; the qmv_fast deep-dive (step 2) and the M1
design doc (step 3) build on this.

All file/line references are to `llama.cpp-SpQt`.

## TL;DR

Two artifacts define the SpQt reference:

1. **`rearrange_tensor_zigzag()`** (`src/llama-quant.cpp:429-446`) — a pure
   **fp-space** rearrangement that runs *before* standard Q4_K quantization.
   Reorders the weight matrix so that when `mx.quantize`-equivalent is then
   applied, **each Q4_K super-block ends up holding 256 output rows at a
   single input column**.

2. **`kernel_mul_mv_q4_K_f32_zigzag_multi_threadgroup_impl()`**
   (`ggml/src/ggml-metal/ggml-metal.metal:5701-5861`) — the dense kernel that
   walks this layout. Per (256-row band × input-column-stride) tile, each TG
   processes one band's contribution to a stride of K-positions, accumulates
   per-row partial sums, then cross-TG-atomic-reduces to produce one
   256-element output slice.

The big idea: **rearrange in fp first, then quantize as usual.** The
quantizer is unchanged — Q4_K still groups 256 contiguous values into one
super-block. The rearrangement just changes *which* values end up adjacent.

## 1. The fp-space rearrangement (`rearrange_tensor_zigzag`)

Source `src/llama-quant.cpp:429-446`:

```cpp
void rearrange_tensor_zigzag(const std::vector<float>& src,
                             std::vector<float>& dst,
                             int n_rows, int n_columns) {
    int superblock_size = 256;                                     // = Q4_K's QK_K
    assert(n_rows % superblock_size == 0);

    int superblock_count = n_rows / superblock_size;
    int superblock_id = 0;
    for (int i = 0; i < superblock_count; i++) {                   // out-band index
        for (int j = 0; j < n_columns; j++) {                      // input column
            for (int k = 0; k < superblock_size; k++) {            // out-row within band
                int src_index = i * n_columns * superblock_size
                                + k * n_columns + j;
                int dst_index = superblock_id * superblock_size + k;
                dst[dst_index] = src[src_index];
            }
            superblock_id += 1;
        }
    }
}
```

### What the indices mean

- `n_rows` = output features (rows of W in (out, in) convention)
- `n_columns` = input features (columns)
- `superblock_size = 256` = Q4_K's super-block size (`QK_K`)

Decoding the source index:

```
src_index = (i * 256 + k) * n_columns + j
```

This is the `(row=i*256+k, col=j)` element of the row-major source tensor. So
the loop traverses each `(out_band=i, in_col=j)` pair, picks up 256 output
rows worth of values at that one input column, and writes them contiguously
to `dst`.

Decoding the destination index (`superblock_id` increments inside the `j`
loop, so `superblock_id = i * n_columns + j`):

```
dst_index = (i * n_columns + j) * 256 + k
```

A 3D layout `[i, j, k]` in row-major order:
- `i` ∈ `[0, n_rows/256)` — which 256-row band of output features
- `j` ∈ `[0, n_columns)` — which input column
- `k` ∈ `[0, 256)` — which output row within the band

### What this layout achieves

In the standard layout, **W[i*256+k, j]** lives at byte offset
`(i*256+k) * n_columns + j` — meaning "all 256 rows' values at one input
column" are *strided* (one element every `n_columns` bytes). Skipping a
sparse input column requires 256 separate loads, mostly wasted.

In the zigzag layout, **W[i*256+k, j]** lives at byte offset
`(i * n_columns + j) * 256 + k` — meaning "all 256 rows' values at one input
column" are now *contiguous* (256 consecutive values). Skipping a sparse
input column means skipping a single contiguous 256-element region.

That contiguity is the entire point.

### Quantization on top

Right after `rearrange_tensor_zigzag`, the standard Q4_K quantizer runs on
`dst` (`src/llama-quant.cpp:1452-1453`):

```cpp
rearrange_tensor_zigzag(data_tmp, dst_rearranged_data, tensor->ne[1], tensor->ne[0]);
new_size += llama_tensor_quantize_impl(new_type, dst_rearranged_data.data(), ...);
```

Q4_K's quantizer always groups 256 contiguous fp values into one super-block.
The rearrangement is structured so that *each contiguous 256-element block*
in `dst` is exactly one `[i, j, *]` slice — 256 output rows at one input
column.

So **after quantize, each Q4_K super-block represents 256 output rows × 1
input column**. The quantizer itself is unchanged; only its input is
reshuffled.

### Memory layout summary

After both steps:

```
quantized_weights (Q4_K)
  ├─ (out_band=0,  in_col=0)    ← super-block 0:  output rows  0..255 at input col   0
  ├─ (out_band=0,  in_col=1)    ← super-block 1:  output rows  0..255 at input col   1
  ├─ ...
  ├─ (out_band=0,  in_col=K-1)  ← super-block K-1: output rows 0..255 at input col K-1
  ├─ (out_band=1,  in_col=0)    ← super-block K:   output rows 256..511 at input col 0
  ├─ (out_band=1,  in_col=1)
  ├─ ...
```

Total super-blocks = `(M / 256) × K`. Each is 144 bytes (Q4_K's per-super-block size).

To skip "input column j" entirely (sparsity at K=j), every super-block of the
form `(*, j, *)` is skippable — strided in memory by `K × 144 bytes` per
band. Within a band, the super-blocks at column j are *single contiguous
144-byte regions*, so the kernel's "load this super-block" memory access for
column j becomes one skipped read per band.

## 2. The dense GEMV kernel (`_zigzag_multi_threadgroup_impl`)

Source `ggml/src/ggml-metal/ggml-metal.metal:5701-5861`. Walking through it.

### Threadgroup geometry

```
sgcpr = 2          // simdgroups per threadgroup
tgcpr = 32         // threadgroup count per (band × activation column)
```

So each (256-row band × input column) is processed by **32 TGs × 2 SGs = 64
simdgroups** working in parallel on a stride of K-positions.

```
sbcptg = ceil(ne00 / (sgcpr * tgcpr))
       = ceil(K / 64)            // super-blocks (= K-positions) per simdgroup
```

For K=4096: `sbcptg = 64` super-blocks per simdgroup.

Grid dispatch:
- `tgpig.x = r0` → which 256-row band (`r0` ∈ `[0, M/256)`)
- `tgpig.y = r1` → combined index encoding (activation_col_idx, tg_within_band)
  - `tgidxr = sgitg + sgcpr * (r1 % tgcpr)` → which K-stride within band
  - `tgidxac = r1 / tgcpr` → activation column (typically 1 for decode)
- `tgpig.z = im` → batch dim (ignored for our M=1 case)

### Work distribution within a simdgroup

```
ix = tiisg / 8        // 0..3 — which sub-stride within the simdgroup
it = tiisg % 8        // 0..7 — position within the sub-stride
iq = it / 4           // 0 or 1 — which "half" of a super-block
ir = it % 4           // 0..3 — position within the half
```

The `tiisg/8` split mirrors the baseline `kernel_mul_mv_q4_K_f32_impl` —
each simdgroup of 32 threads is divided into 4 sub-strides of 8 threads
each. Each sub-stride processes super-blocks at a stride-4 cadence:

```cpp
for (int ib = ix; ib < bound; ib += 4) {  // each thread processes ib = ix, ix+4, ix+8, ...
    float ycurr = y4[ib];                  // ONE activation value per super-block
    // ... extract Q4_K scales (kmask logic) ...
    for (short i = 0; i < 4; ++i) {
        sumf[i*2 + 0]  += ycurr * (q1[i] & 0x000F) * dalls[0] - dmins[0];
        sumf[i*2 + 1]  += ycurr * (q1[i] & 0x0F00) * dalls[0]/256  - dmins[0];
        sumf[i*2 + 8]  += ycurr * (q1[i] & 0x00F0) * dalls[1]/16   - dmins[1];
        sumf[i*2 + 9]  += ycurr * (q1[i] & 0xF000) * dalls[1]/4096 - dmins[1];
        sumf[i*2 +16]  += ycurr * (q2[i] & 0x000F) * dalls[2]      - dmins[2];
        sumf[i*2 +17]  += ycurr * (q2[i] & 0x0F00) * dalls[2]/256  - dmins[2];
        sumf[i*2 +24]  += ycurr * (q2[i] & 0x00F0) * dalls[3]/16   - dmins[3];
        sumf[i*2 +25]  += ycurr * (q2[i] & 0xF000) * dalls[3]/4096 - dmins[3];
    }
}
```

**Key inversion compared to the baseline kernel:**

| Aspect | Baseline `q4_K_f32_impl` | Zigzag `_zigzag_multi_threadgroup_impl` |
|---|---|---|
| What's loaded once per super-block iteration | 32 activation values (`yl`, `yh`) — one input column range | **One activation value** (`ycurr = y4[ib]`) — one input column |
| What's accumulated | 4 partial sums (`sumf[nr0]`) — one per output row in this thread's row stripe | **32 partial sums** (`sumf[32]`) — one per output row in this thread's row stripe |
| What 32 nibbles in a super-block represent | 32 K-positions of one row | **32 output rows at one K-position** |

**This is the layout flip in code form.** The baseline reads many activations and few row partials; zigzag reads one activation per super-block and many row partials. Both are valid GEMV decompositions; zigzag's makes input-column sparsity exploitable because each super-block corresponds to one and only one input column.

### Per-thread output mapping

`sumf[32]` per thread × 32 threads × 2 simdgroups = 2048 partial sums per TG. But the band is only 256 rows wide. So multiple threads contribute to the same output row, requiring within-TG reduction.

The mapping is encoded by `base_idx = 64*iq + 8*ir`:
- `iq` ∈ {0, 1} splits the 256-row band into halves of 128
- `ir` ∈ {0, 1, 2, 3} further splits each half into quarters of 32
- `base_idx` ∈ {0, 8, 16, 24, 64, 72, 80, 88, 128, 136, 144, 152, 192, 200, 208, 216}

Each thread's `sumf[0..7]`, `sumf[8..15]`, `sumf[16..23]`, `sumf[24..31]` map to four 8-wide row stripes at offsets `base_idx`, `base_idx+32`, `base_idx+128`, `base_idx+160` within the band.

### Cross-simdgroup reduction (within TG)

Threadgroup-shared atomic memory `shared_mem[256]` (one int per band row):

```cpp
// initialize shared_mem to 0
for (int i = 0; i < 8; i++) {
    atomic_store_explicit(&shared_mem[i*32 + tiisg%32], 0, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// each thread atomic-adds its 32 partial sums into shared_mem at its base_idx
for (int row = 0; row < 4; row++) {
    int tmp_idx = tmp_base_idx[row];
    atomic_fetch_add_explicit(&shared_mem[tmp_idx + 0], (int)(sumf[row*8 + 0] * SCALE_FACTOR), …);
    // … 8 atomic-adds per row stripe …
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

**Note the `int` casting via `SCALE_FACTOR = 1e5`.** Metal's `atomic_int` has hardware atomics; `atomic_float` is software-emulated and (per their choice here) avoided. They scale floats by 1e5, atomic-add as ints, then unscale at the final read. Cost: ~5 decimal digits of precision.

### Cross-TG reduction (device-side)

Device-level atomic arrays `atomic_array[band × K]` plus `atomic_counter[band]`:

```cpp
if (sgitg == 0) {
    // SG 0 of each TG is responsible for atomic-adding the within-TG sum to the device array
    for (int i = 0; i < 8; i++) {
        int atomic_value = atomic_load_explicit(&shared_mem[(tiisg%32)*8 + i], …);
        atomic_fetch_add_explicit(&atomic_array_cur[(tiisg%32)*8 + i], atomic_value, …);
    }
    if (tiisg == 0) atomic_fetch_add_explicit(atomic_counter_cur, 1, …);

    // last TG to arrive (counter == tgcpr) writes the final result to dst
    int tmp_counter = atomic_load_explicit(atomic_counter_cur, …);
    if (tmp_counter == tgcpr) {
        for (int i = 0; i < 8; i++) {
            int atomic_value = atomic_load_explicit(&atomic_array_cur[(tiisg%32)*8 + i], …);
            dst_f32[(tiisg%32)*8 + i] = (float)atomic_value / SCALE_FACTOR;
        }
    }
}
```

Two-stage cross-TG reduce:
1. Each TG atomically adds its within-TG sum to `atomic_array_cur` (device memory, persistent across TGs).
2. A counter tracks how many TGs have contributed. The TG that brings the counter to `tgcpr` (= 32, total TG count per band) writes the final result to `dst`.

The "last TG writes" pattern avoids needing an explicit reduction kernel after the matmul. **Note:** this requires the host to pre-zero `atomic_array` and `atomic_counter` before each call. (In MLX, we'd need to do this in `eval_gpu` or pre-allocate clear buffers.)

## 3. Adapting to MLX-affine

Two deltas from Q4_K:

### Delta 1 — flat per-group scales (no super-block hierarchy)

Q4_K has hierarchical scales: a fp16 super-block scale (`d`), a fp16 super-block min (`dmin`), and 8 sub-block scales/mins packed into 6-bit chunks.

The kernel decodes them (lines 5775-5794):
```cpp
sc16[0] = sc[0] & kmask1;     // 6-bit sub-block scale 0
sc16[1] = sc[2] & kmask1;     // 6-bit sub-block scale 2
// ... etc.
dalls[i] = dall * sc8[…];     // dall * (sub-block scale)
dmins[i] = dmin * sc8[…];     // dmin * (sub-block min)
```

For MLX-affine: each group of 64 weights has **one fp16 scale and one fp16 bias**, no hierarchy. So:
- Drop the `kmask1/2/3` decoding entirely.
- Drop the 4-way `dalls`/`dmins` arrays (we only have one scale + one bias per group).
- The dequant math simplifies: `weight = scale * q + bias` directly, no sub-block intermediate.

### Delta 2 — group_size 64 vs super-block 256

Q4_K's super-block holds 256 weights. MLX-affine's group holds 64 weights. So our "zigzag block" should be 64 (not 256) elements.

This implies smaller bands: instead of 256-row bands, **64-row bands** for our `group_size = 64` choice (per Working Principle #4, Scope decision #5). Note: in zigzag, the row-band size **is** the quantization group size by construction — same as the reference, where `superblock_size = QK_K = 256` plays both roles in Q4_K.

The fp-rearrangement adapts:

```cpp
// pseudocode for our adapted zigzag rearrangement
int group_size = 64;                                  // was 256 (Q4_K's QK_K)
for (int i = 0; i < M / group_size; i++) {            // out-band index
    for (int j = 0; j < K; j++) {                     // input column
        for (int k = 0; k < group_size; k++) {        // out-row within band
            dst[(i * K + j) * group_size + k] = src[(i * group_size + k) * K + j];
        }
    }
}
```

After this, `mx.quantize(dst, group_size=64, bits=4)` groups every 64
contiguous values, and **each group corresponds to 64 output rows × 1 input
column** — the M-equivalent of Q4_K-SpQt's super-block-per-input-column
structure.

Output shapes:
- `packed_w_zz`: `(M/64, K, 8)` uint32 (each 64-element group = 8 uint32s)
- `scales_zz`, `biases_zz`: `(M/64, K)` fp16

Total bytes: `(M/64) × K × (32 + 2 + 2) = (M/64) × K × 36` (≡ 4.5 bits/weight, same as MLX-affine's flat layout).

### Delta 3 — kernel walk simplifies

Without Q4_K's hierarchical scales, the kernel inner loop becomes simpler.
Per K-position (per group):
- Load 1 activation (`x[k]`).
- Load 8 uint32s = 64 nibbles = 64 output rows' weights at this K-position.
- Load 1 fp16 scale, 1 fp16 bias.
- For each row: `partial[row] += x[k] * (scale * nibble[row] + bias)`.

Distributed across 32 lanes of a simdgroup: each lane handles 64/32 = 2 rows.
Each lane accumulates 2 partials per K-position.

For 4 simdgroups in a TG split into K-strides: each simdgroup walks ~K/(num_TG * num_SG) K-positions and accumulates partials for the same 64 rows.

Within-TG reduction: each simdgroup ends with 64 row-partials. Cross-simdgroup
reduce via threadgroup memory (M2 smoke #4 verified this).

Cross-TG reduction: same atomic-add to device memory + atomic counter pattern
as the Q4_K reference. **Difference**: we can use `atomic_fetch_add_explicit`
on `device atomic_float*` directly (M0b smoke #3 verified this scales to 256
TGs × 32 threads with no loss). No need for the int-scaling workaround the
Q4_K reference uses. **One precision concern saved.**

### Delta 4 — `dispatch_threadgroups` setup

For Q4_K (band size = `superblock_size` = 256):
- Per band: 32 TG × 2 SG × 32 lanes = 2048 threads
- For M=4096, K=4096: 16 bands × 32 TGs/band = 512 TGs total

For MLX-affine (band size = `group_size` = 64):
- Per band: should be smaller; maybe 8 TG × 2 SG × 32 lanes = 512 threads/band? Or fewer?
- For M=4096, K=4096: 64 bands × N TGs/band

The exact threadgroup count per band is a tuning knob. Rough target: total
threads ≈ enough to saturate the GPU (M-series typically 2-4K threads max in
flight). For our shape, ~512-1024 TGs total seems reasonable. Specific number
to be settled in M3 tuning; M1's correctness target should work for any
choice.

### Delta 5 — host-side initialization

The reference kernel relies on `atomic_array` and `atomic_counter` being
**pre-zeroed before each call**. In our `eval_gpu`, we'd need to either:

- Allocate these as `mx::array`s with explicit zero-fill (e.g., a `mx::zeros`-style helper or a separate Metal kernel), OR
- Do `mx::allocator::malloc` and trust it returns zeroed memory (we observed this empirically in M0b but it's not documented behavior).

Cleaner: add a small zero-fill kernel that runs before the main GEMV. Or use
Metal's `MTLBlitCommandEncoder` to memset. Both are M1-implementation details.

## 4. Open questions for the qmv_fast deep-dive (step 2)

1. **Does qmv_fast's `qdot` consumption pattern fit our zigzag layout?**
   - `qdot<T, 16, 4>` consumes 16 weights from a contiguous packed buffer.
   - In zigzag, 16 contiguous weights = 16 *output rows* at one K-position (not 16 K-positions of one row).
   - This is an **inversion** — qdot's math works with either pattern, but
     the role of `x_thread` flips: in the row-tiled case, `x_thread` is 16
     activations at sequential K-positions; in zigzag, `x_thread` would have
     to be 16 copies of the same activation (since all 16 weights see the
     same `x[j]`).
   - Or: rethink whether we can use qdot at all.

2. **What's the right threadgroup geometry given our band size = `group_size` = 64?**
   - Reference uses (32 TG × 2 SG) per band of 256.
   - For our band of 64, scale down: e.g., (16 TG × 2 SG)? Or change SG count?
   - Need to weigh: total work per band vs. atomic-contention cost (cross-TG
     atomic-adds scale with TG count).

3. **Can we use `device atomic_float*` directly for cross-TG reduce?**
   - Yes — verified by M0b smoke #3 at scale (256 TGs × 32 threads, exact).
   - Means we skip Q4_K reference's int-scaling workaround. M1 simplifies.

4. **How does scale/bias broadcasting work in our layout?**
   - Reference: each Q4_K super-block has its own (super-block scale + 8
     sub-block scales/mins). Decode-once per super-block.
   - Ours: each MLX-affine group has *one* fp16 scale + *one* fp16 bias.
     Single load per group; no decode.
   - Should make the kernel inner loop tighter than the reference.

These all get answered in step 2.

## 5. Summary

The SpQt zigzag idea is structurally simple:

> **Rearrange in fp first, then quantize as usual.**

The rearrangement makes "all `group_size` output rows at one input column" a
contiguous run, so when the standard quantizer groups consecutive elements,
each group naturally represents one input column's contribution to a
`group_size`-row band. Sparse input columns then map to skippable contiguous
memory. Row-band size and quantization group size are inherently the same
quantity in this layout — no separate row-tile parameter is needed.

The kernel walks this layout per (band × K-stride), accumulating per-row
partial sums, with two-level reduction (within-TG via threadgroup atomics,
cross-TG via device atomics).

For our MLX-affine port:
- Same rearrangement idea, with band size = `group_size` = 64 (MLX-affine's
  flat group size) instead of 256 (Q4_K's super-block size).
- Flat per-group scales (no Q4_K hierarchy), simpler inner loop.
- Real `atomic_float` cross-TG reduce (no int-scaling workaround).
- Threadgroup geometry to be set in step 2 of M1 design.

Step 2 next: deep-dive `qmv_fast_impl` to identify which parts mirror this
structure and which parts diverge, including the qdot-fits question.
