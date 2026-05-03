# M1 — Zigzag layout + dense GEMV kernel design

The implementation spec for M1. Combines the SpQt-reference layout
(`mlx-spqt-reference.md`) with the qmv_fast structural template
(`mlx-qmv-fast.md`) to produce a concrete plan for `zigzag_quantize` and
the dense zigzag-GEMV Metal kernel.

After review/sign-off, this doc is the contract step 3.4 (`zigzag_quantize`)
and step 3.5 (kernel) build against.

## Goal

Produce a registered MLX custom op `mlx_spqt.dense_zigzag_qmv(x, w_zz, scales_zz, biases_zz)`
that computes the same logical math as `mx.quantized_matmul(x, w_q, scales, biases, ...)`
but on zigzag-laid-out weights. Plus a Python helper
`mlx_spqt.zigzag_quantize(w_fp, group_size, bits)` that produces
`(w_zz, scales_zz, biases_zz)` from the original full-precision weights.

**Single shape**: M=4096, K=4096, fp16 acts, group_size=64, bits=4.

**Done criterion**: `(y_zz - y_ref).abs().max() < 1e-3`, where `y_ref` is the
dequantize-then-fp-matmul reference.

> **A note on row-band size.** A "row-band" in zigzag is a contiguous group of
> output rows that share one quantization group at each K-position. **By
> construction of the zigzag layout, the row-band size equals `group_size`** —
> the rearrangement makes `group_size` rows × 1 K-position contiguous in
> memory, so when the standard quantizer groups along the last axis it picks
> up exactly that slice. This matches llama.cpp-SpQt where `superblock_size =
> QK_K = 256` plays both roles in Q4_K. Throughout this doc we use
> `group_size` for both meanings — there is no separate `M_TILE` parameter.

## Locked decisions

| Decision | Value | Why |
|---|---|---|
| Threadgroup pattern | SpQt-reference-style: multiple SGs per TG split K-walk; within-TG reduce sums per-row partials across SGs | Higher per-TG occupancy (4 SG vs. 2 in qmv_fast-style), lower atomic contention (4 TGs/band vs. 8), better data locality per SG. Reuses M0b smoke #4's threadgroup-reduce pattern. |
| Threadgroup geometry | 4 SG × 32 lanes = 128 threads/TG; `rows_per_lane` = `group_size / SIMD_SIZE` = 2; 4 TGs per band | Mirrors llama.cpp-SpQt's per-TG SG count. `num_simdgroups` and `TGs_per_band` are M3-tunable knobs. |
| `qdot` reuse | **No** — write custom inner loop | Per `mlx-qmv-fast.md` §8: qdot's "16 weights → 1 accumulator" can't service multiple row-partials per K-position |
| Within-TG reduction | `threadgroup atomic<float>` + `atomic_fetch_add_explicit` (Option I); barrier between init/accumulate/read phases | Native MSL atomic on Apple GPU family 7+ (M1+); MLX core's `atomic.h` treats `atomic<float>` as first-class. Cleaner than per-SG disjoint regions + manual sum. |
| Cross-TG reduction | `atomic_fetch_add_explicit` on `device atomic<float>* y` | M0b smoke #3 verified scales lossless to 256 TGs × 32 threads |
| Output zero-fill | Explicit zero-fill in `eval_gpu` before main dispatch | More robust than relying on `mx::allocator::malloc` zeroing (per Phase 6 recommendation) |

## 1. The zigzag layout (concrete)

### Conceptual structure

The original weight matrix `W` has shape `(M, K) = (4096, 4096)`. After
fp-rearrangement, it's logically a 3D tensor `W_rearr` of shape
`(M/group_size, K, group_size) = (64, 4096, 64)` where:

```
W_rearr[band, k_pos, row_in_band] = W[band * group_size + row_in_band, k_pos]
```

- `band` ∈ `[0, M/group_size)` = `[0, 64)` — which group of `group_size` output rows
- `k_pos` ∈ `[0, K)` = `[0, 4096)` — input column
- `row_in_band` ∈ `[0, group_size)` = `[0, 64)` — output row within the band

**The key property**: for a given `(band, k_pos)`, all `group_size` rows' values
are contiguous in memory (the last axis varies fastest in row-major).

### Quantization on this 3D tensor

After rearrangement, `mx.quantize(W_rearr, group_size=64, bits=4)` quantizes
along the last axis. Since the last axis has size = `group_size`, there's
exactly **one quantization group per `(band, k_pos)` slice**.

Each group ⇒ 1 fp16 scale + 1 fp16 bias, shared across all 64 rows in that
slice. **This is the semantic shift from MLX-affine standard**: scales now
vary by `(band, k_pos)` rather than by `(row, k_group)`.

### Concrete shapes for our M=4096, K=4096, gs=64, b=4 case

| Tensor | Shape | dtype | Bytes |
|---|---|---|---|
| `w_zz` | `(M/group_size, K, group_size * bits / 32)` = `(64, 4096, 8)` | uint32 | 64 × 4096 × 8 × 4 = **8 MB** |
| `scales_zz` | `(M/group_size, K)` = `(64, 4096)` | fp16 | 64 × 4096 × 2 = **512 KB** |
| `biases_zz` | `(M/group_size, K)` = `(64, 4096)` | fp16 | 512 KB |
| **Total** | | | **9 MB** (same as MLX-affine standard) |

Note: `mx.quantize` returns `scales_zz` of shape `(64, 4096, 1)` (with the
group dim explicit). We `squeeze(-1)` for cleaner kernel access.

### Memory layout in linear bytes

For `w_zz` (uint32 row-major):

```
byte offset     content
─────────────  ──────────────────────────────────────────────
0          ←  w_zz[0, 0, 0..7]  (band=0, k=0, 64 rows packed into 8 uint32s)
32         ←  w_zz[0, 1, 0..7]  (band=0, k=1, 64 rows packed)
...
4096*32-32 ←  w_zz[0, 4095, 0..7]  (band=0, k=4095, 64 rows packed)
4096*32    ←  w_zz[1, 0, 0..7]  (band=1, k=0, 64 rows packed)
...
```

For `scales_zz` (fp16 row-major):

```
byte offset     content
─────────────  ──────────────────────────────────────────────
0          ←  scales_zz[0, 0]  (band=0, k=0)
2          ←  scales_zz[0, 1]  (band=0, k=1)
...
```

### How a single nibble is reached

Within a `(band, k_pos)` chunk of 8 uint32s, nibble `row_in_band` lives at:
- uint32 index: `row_in_band / 8`
- nibble position within uint32: `row_in_band % 8`
- bits: `[(row_in_band % 8) * 4, (row_in_band % 8) * 4 + 4)` (little-endian within uint32)

So:
```
nibble = (w_zz[band, k_pos, row_in_band / 8] >> (4 * (row_in_band % 8))) & 0xF
```

This matches MLX-affine's standard nibble layout (per `mlx-affine-quantization.md` §3).

## 2. `zigzag_quantize` design

Pure Python/MLX. ~10 lines.

```python
def zigzag_quantize(w, group_size, bits):
    """Take a full-precision (M, K) weight matrix; return zigzag-quantized form.

    Parallels mx.quantize's signature shape but produces the layout described
    in M1-zigzag-layout.md §1.
    """
    M, K = w.shape
    assert M % group_size == 0, f"M ({M}) must be divisible by group_size ({group_size})"
    assert bits == 4, "MVP supports bits=4 only"

    # Rearrange in fp space: (M, K) → (M/group_size, group_size, K) → (M/group_size, K, group_size)
    # The transpose is what makes group_size rows for one k_pos contiguous in the last axis.
    w_rearr = w.reshape(M // group_size, group_size, K).transpose(0, 2, 1)

    # Quantize along last axis (size = group_size, so one group per slice)
    w_zz, scales_zz, biases_zz = mx.quantize(w_rearr, group_size=group_size, bits=bits)

    # Squeeze the trailing-1 group dim for cleaner kernel access
    scales_zz = scales_zz.squeeze(-1)
    biases_zz = biases_zz.squeeze(-1)

    return w_zz, scales_zz, biases_zz
```

Output shapes match §1's table: `(M/group_size, K, 8)`, `(M/group_size, K)`, `(M/group_size, K)`.

### Verification: dequantize round-trip

For the M1 test, we'll need to dequantize the zigzag form back to a full
`(M, K)` fp tensor. Helper:

```python
def dequantize_zigzag(w_zz, scales_zz, biases_zz, group_size, bits):
    """Dequantize a zigzag-laid-out weight tensor back to full-precision (M, K)."""
    # Re-add the squeezed trailing-1 group dim for mx.dequantize
    scales_with_gd = scales_zz[..., None]  # (M/group_size, K, 1)
    biases_with_gd = biases_zz[..., None]

    # Dequantize: returns shape (M/group_size, K, group_size)
    w_dequant_rearr = mx.dequantize(
        w_zz, scales_with_gd, biases_with_gd, group_size=group_size, bits=bits
    )

    # Un-rearrange: (M/group_size, K, group_size) → (M/group_size, group_size, K) → (M, K)
    n_bands, K, _ = w_dequant_rearr.shape
    w_dequant = w_dequant_rearr.transpose(0, 2, 1).reshape(n_bands * group_size, K)

    return w_dequant
```

This is the inverse of `zigzag_quantize` modulo quantization error.

### Test 3.4 (zigzag_quantize correctness in isolation)

Before the kernel exists, verify the Python helpers:

```python
M, K = 4096, 4096
group_size, bits = 64, 4

w = mx.random.normal(shape=(M, K), dtype=mx.float16)

# Round-trip
w_zz, scales_zz, biases_zz = zigzag_quantize(w, group_size, bits)
w_recon = dequantize_zigzag(w_zz, scales_zz, biases_zz, group_size, bits)

# Verify shape match
assert w_recon.shape == w.shape
# Verify reconstruction error is bounded by quantization noise
err = (w - w_recon).abs().max().item()
print(f"max absolute reconstruction error: {err}")
# Expected: roughly bounded by max(scale) — typical fp16 quant gives errors < 0.1 for normalized weights
assert err < 1.0, "Reconstruction error too large; check rearrangement logic"
```

If this passes, `zigzag_quantize` and `dequantize_zigzag` correctly invert
each other. The kernel can be built against this layout with confidence.

## 3. Kernel architecture

This kernel mirrors **llama.cpp-SpQt's threadgroup pattern** (multiple
simdgroups split the K-walk, all simdgroups in a TG cover the same band's
rows; within-TG reduce sums simdgroup partials; cross-TG atomic merges
TG partials). Choosing this over qmv_fast's "1 lane = 1 row, no within-TG
reduce" pattern because:

- **Higher per-TG occupancy**: 4 simdgroups vs. 2 → better latency hiding.
- **Lower atomic contention**: fewer TGs/band (4 vs. 8) → fewer atomic
  adds per output row.
- **Better data locality**: each simdgroup walks a contiguous K-stripe for
  one band — band's weights stay warm in L1 across the SG's K-walk.
- **Reuses M0b smoke #4** (threadgroup-memory + barrier reduce); this is
  why we built that smoke kernel.

### Threadgroup geometry

| Quantity | Value | Notes |
|---|---|---|
| Simdgroups per TG (`num_simdgroups`) | 4 | matches SpQt reference's per-TG SG count |
| Lanes per simdgroup | 32 | hardware (`SIMD_SIZE`) |
| Threads per TG | 128 | 4 × 32 |
| Rows owned per lane | 2 | `group_size / SIMD_SIZE` = 64/32 |
| Rows per simdgroup | 64 | each SG produces partials for **all** rows of the band |
| Bands handled per TG | 1 | one TG owns one `group_size`-row band |
| TGs per band | 4 | starting recommendation; 4-8 tunable in M3 |
| K-positions per SG | `K / (TGs_per_band × num_simdgroups)` = 4096 / 16 = **256** | each SG walks 256 K-positions |
| K-positions per TG | `K / TGs_per_band` = 4096 / 4 = **1024** | TG covers 1024 K-positions |
| Total TGs | 64 bands × 4 TGs/band = **256** | |
| Total threads | 256 × 128 = **32768** | same total work as Option A; packed differently |
| Threadgroup memory | `group_size × 4 B` = **256 B** | shared `atomic<float>` accumulator (Option I); all SGs atomic-add into the same 64 slots |

### Per-thread work assignment

```
band       = tid.x                                  // ∈ [0, M/group_size) = [0, 64)
tg_in_band = tid.y                                  // ∈ [0, TGs_per_band) = [0, 4)
simd_gid   ∈ [0, num_simdgroups) = [0, 4)
simd_lid   ∈ [0, 32)

// Each lane owns rows {2*simd_lid, 2*simd_lid + 1} within the band, accumulating
// partials over its SG's K-stripe. All 4 SGs accumulate for the SAME 64 rows
// (over different K-stripes), and we reduce across SGs at the end.
rows_owned = {2*simd_lid, 2*simd_lid + 1}           // ∈ [0, group_size)
global_rows = band * group_size + rows_owned        // ∈ [0, M)
```

### K-walk slicing — two-level

```
k_per_tg = K / TGs_per_band                         // 4096 / 4 = 1024
k_per_sg = k_per_tg / num_simdgroups                // 1024 / 4 = 256

k_start  = tg_in_band * k_per_tg + simd_gid * k_per_sg
k_end    = k_start + k_per_sg
```

Each SG covers a 256-K-position contiguous stripe of one band. After the
K-walk, each SG's 32 lanes hold `rows_per_lane = 2` row-partials in registers.
All `num_simdgroups` SGs in a TG hold partials for the same 64 rows but over
disjoint K-stripes; we sum them via **`threadgroup atomic<float>`** (Option I)
— each lane atomic-adds its partials directly into a shared TG-mem
accumulator that all SGs share.

### Grid dispatch

```cpp
constexpr int num_simdgroups = 4;
constexpr int TGs_per_band   = 4;
MTL::Size group_dims(num_simdgroups * SIMD_SIZE, 1, 1);     // 128 threads/TG
MTL::Size grid_dims(M / group_size, TGs_per_band, 1);       // (64, 4, 1) = 256 TGs
```

## 4. Kernel inner loop (replaces qdot)

Per Working Principle #4: **mirror qmv_fast's structural conventions, but
write a custom inner loop** since qdot doesn't fit (per `mlx-qmv-fast.md` §8).
The structural mirror: SG walks a K-stripe → accumulates row-partials in
per-lane registers → reduces across lanes/SGs → writes. The custom inner
loop replaces qdot's "16 weights × 1 row" with "1 weight × `rows_per_lane`
rows" per K-position.

### What runs per K-position (per SG)

For one K-position `k` within a SG's stripe:

1. Compute group index: `k_group = k / group_size`. (For our locked
   `group_size = 64`, every K-position is also a group boundary, so
   `k_group = k`.)
2. Read **shared** scale and bias for this `(band, k_group)`:
   - `scale = scales_zz[band, k_group]`,  `bias = biases_zz[band, k_group]`
   - All 32 lanes in the SG read the same value (broadcast / cached).
3. Read this **lane's uint32** at this K-position:
   - Lane `simd_lid` reads `w_zz[band, k, simd_lid * rows_per_lane / pack_factor]`.
   - 32 lanes collectively read 8 distinct uint32s (each uint32 read by 4 lanes;
     hardware coalesces) = 32 bytes, one cache line.
4. Read the activation: `xk = x[k]`. Single fp16 load, broadcast.
5. Unpack `rows_per_lane` nibbles from the uint32 and accumulate into per-lane
   row-partials:
   ```
   for r in 0..rows_per_lane:
       nibble = (packed >> (4 * nibble_pos)) & 0xF
       partial[r] += xk * (scale * float(nibble) + bias)
   ```

### Pseudocode

```metal
template <typename T, int group_size, int bits, int num_simdgroups, int TGs_per_band>
[[kernel]] void zigzag_qmv_dense_impl(
    device const uint32_t*    w_zz       [[buffer(0)]],
    device const T*           scales_zz  [[buffer(1)]],
    device const T*           biases_zz  [[buffer(2)]],
    device const T*           x          [[buffer(3)]],
    device atomic<float>*     y          [[buffer(4)]],
    constant const int&       K          [[buffer(5)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]) {

  // ─── Constexpr derivations (mirror qmv_fast) ────────────────────────────
  constexpr int pack_factor    = get_pack_factor<bits, 32>();    // 8 for bits=4
  constexpr int rows_per_lane  = group_size / SIMD_SIZE;         // 2 for gs=64

  // ─── Position math ──────────────────────────────────────────────────────
  const int band       = tid.x;
  const int tg_in_band = tid.y;

  const int k_per_tg = K / TGs_per_band;
  const int k_per_sg = k_per_tg / num_simdgroups;
  const int k_start  = tg_in_band * k_per_tg + simd_gid * k_per_sg;
  const int k_end    = k_start + k_per_sg;

  const int wzz_stride_band = K * (group_size / pack_factor);    // = K * 8
  const int wzz_stride_k    = group_size / pack_factor;          // = 8
  const int scl_stride_band = K / group_size;                    // = K / 64

  // Lane simd_lid owns rows {row_lo, row_lo + 1, ..., row_lo + rows_per_lane - 1}
  const int row_lo          = simd_lid * rows_per_lane;
  const int nibble_uint_idx = row_lo / pack_factor;              // ∈ [0, 8)
  const int nibble_base_pos = row_lo % pack_factor;              // ∈ [0, 8)

  // ─── (1) Initialize threadgroup accumulator in parallel ─────────────────
  // All threads in TG participate. Each lane stores 0 into its `rows_per_lane`
  // owned slots. Across SGs, the same slot is initialized num_simdgroups times,
  // but all writes are 0 → race-safe.
  threadgroup atomic<float> tg_partials[group_size];

  #pragma unroll
  for (int r = 0; r < rows_per_lane; r++) {
    atomic_store_explicit(&tg_partials[row_lo + r], 0.0f, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // ── barrier #1: init done before any accumulate

  // ─── (2) Per-lane accumulators (rows_per_lane partials in registers) ────
  thread float partial[rows_per_lane] = {0};

  // ─── (3) K-walk (custom inner loop replacing qdot) ──────────────────────
  for (int k = k_start; k < k_end; k++) {
    const int k_group = k / group_size;
    const float scale = (float)scales_zz[band * scl_stride_band + k_group];
    const float bias  = (float)biases_zz[band * scl_stride_band + k_group];
    const float xk    = (float)x[k];

    const uint32_t packed = w_zz[band * wzz_stride_band + k * wzz_stride_k + nibble_uint_idx];

    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      const uint nibble = (packed >> (4 * (nibble_base_pos + r))) & 0xF;
      partial[r] += xk * (scale * float(nibble) + bias);
    }
  }

  // ─── (4) Within-TG reduce: each lane atomic-adds its partials ───────────
  //   All num_simdgroups SGs target the same 64 slots; atomic_fetch_add
  //   serializes contributions to each slot.
  #pragma unroll
  for (int r = 0; r < rows_per_lane; r++) {
    atomic_fetch_add_explicit(&tg_partials[row_lo + r], partial[r], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);   // ── barrier #2: all atomic-adds visible to SG 0

  // ─── (5) Cross-TG reduce: SG 0 atomic-adds TG totals to global y ────────
  if (simd_gid == 0) {
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      const float sum = atomic_load_explicit(&tg_partials[row_lo + r], memory_order_relaxed);
      const int global_row = band * group_size + row_lo + r;
      atomic_fetch_add_explicit(&y[global_row], sum, memory_order_relaxed);
    }
  }
}
```

### Two-level reduction summary

| Level | Pattern | Mechanism | Smoke verified |
|---|---|---|---|
| Within-SG | (none — each lane owns disjoint rows of its SG's row-partial set) | — | n/a |
| Within-TG (across SGs) | Sum `num_simdgroups` partials per row | `threadgroup atomic<float>` + `atomic_fetch_add_explicit`; two `threadgroup_barrier`s (init→accumulate, accumulate→read) | M0b smoke #4 ✓ |
| Cross-TG (across K-stripes per band) | Sum `TGs_per_band` partials per row | Device `atomic_fetch_add_explicit` on `atomic<float>` | M0b smoke #3 ✓ |

The kernel uses both reduction patterns we built smoke tests for. No new
unverified Metal mechanism is introduced. Native MSL `atomic<float>` is
supported on Apple GPU family 7+ (M1+); MLX core's `mlx/backend/metal/kernels/atomic.h:14-21`
treats `float` as a first-class atomic type alongside `int`/`uint`/`ulong`.

## 5. Memory access analysis

| Access | Pattern | Cost |
|---|---|---|
| Per SG, per K-position: 32 lanes read 8 distinct uint32s (each by 4 lanes) | 32-byte coalesced load (= 1 cache line); HW broadcasts within | 1 ✓ |
| Per SG, per K-position: scale + bias broadcast | All 32 lanes load same fp16 pair → cached after first lane | ~1 |
| Per SG, per K-position: x[k] broadcast | All 32 lanes load same fp16 → cached | ~1 |
| Per SG K-stripe: 256 K-positions sequential in memory (band's contiguous stripe) | Sequential cache-line stream → L1 prefetch friendly | low |
| Threadgroup atomic-add (within-TG reduce) | 4 SGs × 32 lanes × 2 atomic_fetch_add per K-walk-end = 256 ops, contending on 64 slots | small; one barrier on each side |
| Threadgroup memory size | `group_size × 4 B` = 256 B per TG | trivial |

**Bandwidth per K-position per SG**: ~32 bytes weights + ~6 bytes
(scales + biases + x, broadcasted) = ~38 bytes.

**Total bandwidth per band**: 4096 K-positions × 38 bytes ≈ **156 KB per band**,
read once across all `num_simdgroups × TGs_per_band` = 16 SGs covering disjoint
K-stripes (each SG reads K/16 = 256 K-positions × 38 bytes ≈ 9.7 KB).

**Total reads for full GEMV**: 64 bands × 156 KB ≈ **9.8 MB** — matches
the size of `w_zz + scales_zz + biases_zz`. Theoretically optimal: each
input byte read once.

**Atomic contention**: each output row receives `TGs_per_band = 4` atomic
adds (one per TG that covered its band). Half the contention of a 1-lane-per-row
design with 8 TGs/band.

## 6. eval_gpu wrapper

```cpp
void DenseZigzagQMV::eval_gpu(const std::vector<mx::array>& inputs,
                              std::vector<mx::array>& outputs) {
  auto& x         = inputs[0];
  auto& w_zz      = inputs[1];
  auto& scales_zz = inputs[2];
  auto& biases_zz = inputs[3];
  auto& out       = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  // (1) Allocate AND zero-fill output (atomic-add accumulates from initial value)
  out.set_data(mx::allocator::malloc(out.nbytes()));
  zero_fill_buffer(out, s, d);    // explicit zero-fill helper; either MTLBlitCommandEncoder fillBuffer
                                  // or a small zero-fill kernel

  // (2) Look up the kernel
  auto lib = d.get_library("mlx_spqt", current_binary_dir());
  auto kernel = d.get_kernel("zigzag_qmv_dense_half_gs64_b4_NSG4_TG4", lib);

  // (3) Set up the encoder + bind buffers
  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(w_zz, 0);
  enc.set_input_array(scales_zz, 1);
  enc.set_input_array(biases_zz, 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);
  int K = x.shape(-1);
  enc.set_bytes(K, 5);

  // (4) Dispatch
  constexpr int num_simdgroups = 4;
  constexpr int TGs_per_band   = 4;
  constexpr int SIMD_SIZE      = 32;
  const int M = out.shape(-1);
  const int group_size = 64;                                        // matches scales_zz layout
  MTL::Size group_dims(num_simdgroups * SIMD_SIZE, 1, 1);           // 128 threads/TG
  MTL::Size grid_dims(M / group_size, TGs_per_band, 1);             // 64 × 4 = 256 TGs
  enc.dispatch_threadgroups(grid_dims, group_dims);
}
```

The `zero_fill_buffer` helper is the only piece that's new vs. previous
kernels. Two implementation approaches:

- **Approach 1 (preferred)**: a small Metal kernel `zero_fill_float` that
  takes `device float* out` and writes zeros. Dispatches `out.size()` threads.
- **Approach 2**: `MTLBlitCommandEncoder::fillBuffer` — no compute kernel
  needed. Slightly more involved C++ side (different encoder type) but no
  extra MSL.

I'd recommend **Approach 1** for simplicity (mirrors what we already know
from the smoke kernels). MSL and dispatch are trivial.

### Kernel name & instantiation

For M1, single template instantiation:

```metal
template [[host_name("zigzag_qmv_dense_half_gs64_b4_NSG4_TG4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 4, 4>(...);
```

Naming convention: `_NSG4` = num_simdgroups, `_TG4` = TGs_per_band. Future
M3 perf-tuning will instantiate additional `(NSG, TG)` combinations from the
same template body.

## 7. The M1 done criterion (correctness test)

```python
import mlx.core as mx
import mlx_spqt
import numpy as np

M, K = 4096, 4096
group_size, bits = 64, 4

mx.random.seed(42)
w = mx.random.normal(shape=(M, K), dtype=mx.float16)
x = mx.random.normal(shape=(1, K), dtype=mx.float16)

# Step 1: zigzag-quantize
w_zz, scales_zz, biases_zz = mlx_spqt.zigzag_quantize(w, group_size=group_size, bits=bits)

# Step 2: our kernel
y_zz = mlx_spqt.dense_zigzag_qmv(x, w_zz, scales_zz, biases_zz)

# Step 3: reference (dequantize-then-fp-matmul)
w_dequant = mlx_spqt.dequantize_zigzag(w_zz, scales_zz, biases_zz, group_size=group_size, bits=bits)
y_ref = (x @ w_dequant.T).astype(mx.float16)

# Step 4: verify within MLX's standard tolerance
err = (y_zz - y_ref).abs().max().item()
print(f"max absolute error: {err}")
assert err < 1e-3, f"M1 done criterion violated: max err = {err}"
print("M1 dense zigzag-GEMV: PASS")
```

Same predicate as MLX's own `test_quantized.py::test_qmv:355`.

**Why this is the right reference (not stock `mx.quantized_matmul`)**: stock
quantizes row-wise (different scales per row); zigzag quantizes column-wise
(different scales per K-position per band). They produce different
quantization errors against fp32 ground truth. Our test verifies *our
kernel's math is consistent with our quantization scheme*, which is the
correct M1 contract.

## 8. Implementation step-by-step (for step 3.4 + 3.5)

### Step 3.4 — `zigzag_quantize` (Python helper) ✅

Implemented in `extensions/mlx_spqt/mlx_spqt/quantization.py`. Round-trip
test (`extensions/mlx_spqt/test_quantize_zigzag.py`) passes with max abs
error 0.4746, within 3% of standard MLX-affine on the same input.

### Step 3.5 — Dense zigzag-GEMV kernel extension

Mirroring the M0b add-a-kernel workflow:

| Sub-step | Action | Effort |
|---|---|---|
| 3.5.a | Create `extensions/mlx_spqt/zigzag_qmv_dense/` with stub `.h` / `.cpp` / `.metal` | 30 min |
| 3.5.b | Wire into `bindings.cpp` + `CMakeLists.txt` + `__init__.py` | 30 min |
| 3.5.c | Build (`pip install -e . --force-reinstall --no-deps`); verify import gate | 10 min |
| 3.5.d | Implement the Metal kernel body per §4 (including within-TG reduce) | 2-3 h |
| 3.5.e | Implement `eval_gpu` wrapper per §6 (including zero-fill helper) | 1 h |
| 3.5.f | Write `test_dense_zigzag_qmv.py` per §7; iterate until PASS | 1-2 h |

Total: ~5-8 hours.

### Step 3.6 — Final test in `tests/test_spqt.py`

Add a top-level test entry that calls the M1 done-criterion test (per §7).
This becomes the "correctness gate" that M3's bench depends on (only run
perf bench if correctness holds).

## 9. What this design does NOT lock in

- **Threadgroup geometry tuning**: the choice of `(num_simdgroups, TGs_per_band)
  = (4, 4)` is a starting point; M3 tuning may revisit (`TGs_per_band` range:
  2-8; `num_simdgroups` range: 2-8).
- **Whether to use `device atomic_float*` or int-scaling fallback**: we use
  `atomic_float` directly per Phase 6 recommendation. Upstream-merge may
  later add the fallback.
- **Layout for bits ≠ 4 or group_size ≠ 64**: locked to `(bits=4, gs=64)` for
  MVP. Future generalization is straightforward (kernel is already templated
  on these).
- **CPU implementation, autograd, vmap**: all NYI per Scope decision (matches
  `QuantizedMatmul` in MLX core).

## 10. Open questions resolved

From `mlx-spqt-reference.md` §4 + `mlx-qmv-fast.md` §8 + Phase 6:

| Question | Resolution |
|---|---|
| Does qdot fit zigzag layout? | **No** — custom inner loop. Mirror qmv_fast's *structure* (template, accumulator types, simdgroup conventions); replace its inner call. |
| qmv_fast-style ("1 lane = 1 row, no within-TG reduce") or SpQt-reference-style ("multiple SGs split K, within-TG reduce")? | **SpQt-reference-style.** Higher per-TG occupancy (4 SG vs. 2), lower atomic contention (4 TGs/band vs. 8), better data locality per SG, structural reuse from qmv_fast still preserved (template, register-resident partials, K-walk shape). Reuses M0b smoke #4's threadgroup-memory reduce pattern. |
| Threadgroup geometry for `group_size`=64? | 4 SG × 32 lanes = 128 threads/TG. Each lane owns 2 rows of the band; all 4 SGs cover all 64 rows, splitting K. Within-TG reduce sums per-row partials across SGs. 4 TGs per band as starting point; tunable in M3. |
| Cross-TG `atomic_float` directly? | **Yes** (M0b smoke #3 verified). |
| Within-TG reduce mechanism? | `threadgroup atomic<float>` + two `threadgroup_barrier`s (Option I). Native MSL on Apple GPU family 7+ (M1+); MLX `atomic.h` treats `atomic<float>` as first-class. 256 B threadgroup memory per TG. M0b smoke #4 verified the underlying TG-mem + barrier pattern. |
| Single-scale broadcasting? | Yes — all 32 lanes in an SG read the same `(scale, bias)` pair per K-position (one quant group per `(band, k_pos)` slice). Coalesces naturally via cache. |
| Row-band size vs. `group_size`? | They are **the same quantity** by construction of zigzag. The rearrangement makes `group_size` rows × 1 K-position contiguous; the standard quantizer then picks up exactly one group per `(band, k_pos)` slice. No separate `M_TILE` parameter. |
| Pre-zero output for atomic-add? | Yes — explicit zero-fill kernel in `eval_gpu` before the main dispatch. |

## 11. As-built kernel — deviations from the planned design

The M1 kernel is **landed and correctness-verified**, but the implementation
took two pivots from the design above. Documenting them here so the
plan-vs-reality gap is explicit.

### Pivot 1 — within-TG reduce mechanism

**Planned**: `threadgroup atomic<float>` + two `threadgroup_barrier`s (Option I).

**Reality**: `threadgroup atomic<float>` **is not supported on Metal** — float
atomics work in `device` address space only. MSL's `_valid_store_type`
allow-list excludes `threadgroup float*` for atomic ops; only `int`/`uint`/
`long`/`ulong` work in threadgroup atomic operations. (This is also why
llama.cpp-SpQt uses `threadgroup atomic_int` with a `SCALE_FACTOR` for its
within-TG reduce: it's the only float-equivalent that compiles.)

**As built**: **No within-TG reduce at all.** Each lane atomic-adds its
per-row partials directly to `device atomic<float>* y`. With 8 lanes per
row chunk × 4 SGs × 4 TGs = 128 atomic-adds per output row per kernel
launch. Higher cross-TG atomic contention than the planned design, but
correct, and substantially simpler (no threadgroup memory, no barriers, no
scaling tricks).

### Pivot 2 — per-thread work assignment

**Planned**: 1 lane = `rows_per_lane = group_size / SIMD_SIZE = 2` rows; outer
loop iterates K_per_sg = 256 K-positions.

**As built**: 1 lane = `values_per_thread = 16` rows per outer iter; lanes are
arranged as an 8 × 4 (K-position × row-chunk) grid within each SG; outer
loop iterates 32 times (each lane covers 32 distinct K-positions).

**Lane assignment**:

```
                K-pos 0   K-pos 1   K-pos 2   K-pos 3   K-pos 4   K-pos 5   K-pos 6   K-pos 7
row chunk 0     lane 0    lane 4    lane 8    lane 12   lane 16   lane 20   lane 24   lane 28
row chunk 1     lane 1    lane 5    lane 9    lane 13   lane 17   lane 21   lane 25   lane 29
row chunk 2     lane 2    lane 6    lane 10   lane 14   lane 18   lane 22   lane 26   lane 30
row chunk 3     lane 3    lane 7    lane 11   lane 15   lane 19   lane 23   lane 27   lane 31
```

K-position index = `simd_lid / 4`; row-chunk index = `simd_lid % 4` (or
equivalently `simd_lid * values_per_thread % group_size / values_per_thread`).
Each lane owns 16 rows × 32 K-positions over the full outer loop.

This is closer to a qmv_fast-style "many K-positions per thread" pattern,
adapted for zigzag's row-chunk structure. The inner loop loads 1
`(x, scale, bias)` per outer iter (one K-position) and unpacks 16 nibbles
from 4 uint16 reads (the lane's 16 rows at that K-position).

### What's deferred for future optimization

The current kernel is a **working draft** — correct but not optimized:

- **Within-TG reduce**: would cut atomic contention 8× (lanes within an SG
  contributing to the same row-chunk could pre-sum via threadgroup memory
  before the cross-TG atomic). Not strictly needed for correctness; the
  llama.cpp-SpQt int+scaling pattern would work, or Option II's per-SG
  disjoint-region pattern.
- **Within-SG `simd_sum` reduce**: in our current geometry, 4 lanes per
  K-pos compute partials for 4 distinct row chunks (no within-SG redundancy).
  But across SGs, lanes 0/4/8/12/... all hit the same row chunk — that's
  where cross-SG reduction would help.
- **Optimal `(num_simdgroups, TGs_per_band)`**: M3 perf-tuning territory.
- **`load_vector`-style activation amortization**: not used; could load a
  K-block of activations once and reuse across multiple K-positions.
- **Multi-batch (B>1)**: kernel signature doesn't include batch stride; would
  need `out_vec_size`-equivalent and a batch dim in the grid.

### Done criterion verification

Test `extensions/mlx_spqt/test_zigzag_qmv_dense.py` passes with `err < 1e-3`.
Specifically: `max abs err = 9.2e-5` against an fp32-precision reference
(scales/biases cast to fp32 before `dequantize_zigzag` to avoid fp16
truncation in the reference itself; the original fp16-truncated reference
would show 0.047 absolute error, dominated by the reference's own fp16
precision loss, not kernel error).

## 12. Implementation experience — lessons captured

Six lessons worth carrying forward, including the false starts:

### A. `threadgroup atomic<float>` is a Metal hard constraint

Float atomics in MSL are device-address-space only. Apple GPU family 7+
gates float atomic *fetch_add* even for device space; threadgroup space is
restricted to integer atomic types. **Always check `nm -g` against
`libmlx.dylib` for symbol exports** before assuming an MLX backend helper
is callable from extensions:

```bash
nm -g python/mlx/lib/libmlx.dylib | grep <symbol>
# Capital T = exported, lowercase t = internal/hidden
```

We discovered `mlx::core::fill_gpu` is hidden the hard way after writing
code that linked against it.

### B. C++ default arguments — header XOR cpp, not both

Default values for function parameters can only appear in **one**
declaration. Convention: defaults in the header (the public interface),
none in the implementation file. We hit "redefinition of default argument"
twice for the same op signature.

### C. MSL position attributes need consistent dimensionality

`[[threadgroup_position_in_grid]]` and `[[thread_position_in_threadgroup]]`
are both *position* attributes — if one is `uint3`, both must be `uint3`
(or both scalar). Index attributes (`[[simdgroup_index_in_threadgroup]]`,
`[[thread_index_in_simdgroup]]`) are scalar regardless. Mixing
`uint3 tid` with `uint thread_id` triggers "expecting input declarations
with either all scalar types or all vector types" — no obvious fix from
the error message itself.

### D. Pointer types must match between template and host_name instantiation

```metal
template <...> [[kernel]] void zigzag_qmv_dense_impl(
    device const uint32_t*  w_zz [[buffer(0)]], ...);

template [[host_name("...")]]
[[kernel]] void zigzag_qmv_dense_impl<half, ...>(
    device const uint32_t*  w_zz [[buffer(0)]], ...);    // ← MUST match (incl. *)
```

A missing `*` in the instantiation produces "explicit instantiation does
not refer to a function template" — the actual cause is type mismatch in
the parameter list, which the compiler reports indirectly.

### E. 2D lane assignment needs both `simd_lid / X` and `simd_lid % X`

When N lanes are tiled into an X × Y grid (where N = X × Y), every
per-lane offset (read pointer, write index, etc.) needs to encode both
`simd_lid / X` (column dimension) and `simd_lid % X` (row dimension).
qmv_fast gets away with 1D indexing because all lanes share rows; only K
varies. Zigzag's "lanes split row chunks AND K-positions" is genuinely
2D, so missing one dimension caused systemic data corruption — lanes 1-3
read row-chunk-0's nibbles but wrote them to row-chunks 1-3.

A useful diagnostic: **whenever you see one lane stride in reads and a
different lane stride in writes, that's almost always a divmod-decomposition
bug**.

### F. Choose your reference's precision deliberately

The first time the kernel "passed all known design checks" but the test
showed max-err 0.047, the kernel was actually correct. The test's
reference was casting `dequantize_zigzag(...)` (fp16) to fp32 — but the
fp16 truncation had already happened. Casting *scales/biases* to fp32
before `dequantize_zigzag` produces an fp32-precision reference and shrinks
the err to 9.2e-5.

For a kernel that internally computes in fp32 from fp16 inputs, the
reference must mirror that promotion to be a fair correctness check.
Otherwise you're measuring the reference's precision, not the kernel's
correctness.

## Summary

M1 design is locked. The implemented kernel mirrors **llama.cpp-SpQt's threadgroup
pattern**: 4 simdgroups × 32 lanes per TG split a band's K-walk; within-TG
reduce sums per-row partials across simdgroups; cross-TG atomic merges
TG-level partials. From qmv_fast we keep the template signature, register-
resident accumulator pattern, and helper plumbing. From the SpQt reference
we adopt the multi-SG-split-K geometry plus the two-level reduction. The
qdot inner is replaced by a custom dequant-and-FMA loop (per K-position:
load 1 scale + 1 bias + 1 activation + 1 uint32, unpack `rows_per_lane`
nibbles, FMA into per-lane row-partials).

`zigzag_quantize` is a 10-line Python helper, already implemented and
verified. Kernel implementation (step 3.5) is ~5-8 hours of bounded work
now that the design is locked.
