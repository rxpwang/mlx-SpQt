// Copyright © 2023-2025 Apple Inc.

///////////////////////////////////////////////////////////////////////////////                            
// Zero-fill kernel — used by eval_gpu to zero-init y before the main kernel
// (cross-TG atomic-add accumulates from initial value, so initial must be 0)                              
///////////////////////////////////////////////////////////////////////////////                            
                                                                                                            
[[kernel]] void zigzag_zero_fill_float(
    device float* out [[buffer(0)]],
    uint id [[thread_position_in_grid]]) {
  out[id] = 0.0f;
}


///////////////////////////////////////////////////////////////////////////////
// Sparse-indexing kernel — compacts the positions of |x| > tau into a
// count-prefix int32 buffer [n, idx_0, ..., idx_{n-1}] without ever leaving
// the GPU. Replaces the `argsort + concatenate` pattern that dominated
// per-call overhead in the SparseHookedLinear integration.
//
// Design ported from llama.cpp-SpQt's `kernel_sparse_indexing_v2`
// (ggml-metal.metal:2057):
//   - simd_prefix_exclusive_sum gives each lane its compacted slot within the SG.
//   - Per-SG totals tree-reduce to a TG total via threadgroup memory.
//   - Multi-TG case: cross-TG atomic counter for base offset; the last TG
//     to finish writes the final count to slot 0.
//   - One pass over K, no sort.
//
// Buffer layout (caller-allocated, pre-zeroed):
//   dst:           int32 buffer of size K + 1 — receives [count, idx_0, ...]
//   atomic_count:  int32 buffer of size 2 — slot 0 = running offset, slot 1 = TGs finished
///////////////////////////////////////////////////////////////////////////////

template <typename T>
[[kernel]] void zigzag_sparse_indexing_v2_impl(
    device const T*       x                  [[buffer(0)]],
    device int*           dst                [[buffer(1)]],
    constant const int&   K                  [[buffer(2)]],
    constant const float& threshold          [[buffer(3)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]) {

    // Single-TG variant: handles K up to 4 × ntg by looping over chunks
    // sequentially within one threadgroup. Avoids the multi-TG race
    // (Metal doesn't expose acq_rel atomics) at the cost of serial chunk
    // processing. For CD shapes (K ≤ 3072) this is 1-3 chunks.
    constexpr int ntg = 1024;
    constexpr int sgcpr = ntg / 32;

    threadgroup int group_totals[32];
    threadgroup int group_bases[32];
    threadgroup int tg_total_count;
    threadgroup int running_total;

    if (tiitg == 0) {
        running_total = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int num_chunks = (K + ntg - 1) / ntg;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * ntg;
        const int chunk_end = min(chunk_start + ntg, K);
        const int chunk_size = chunk_end - chunk_start;

        // Step 1: per-lane flag for this chunk
        int flag = 0;
        if ((int)tiitg < chunk_size) {
            float val = (float)x[chunk_start + (int)tiitg];
            flag = (val > threshold || val < -threshold) ? 1 : 0;
        }

        // Step 2: SIMD prefix sum
        int simd_prefix = simd_prefix_exclusive_sum(flag);
        int simd_total = simd_sum(flag);

        // Step 3: write per-SG totals to TG memory
        if (tiisg == 0) {
            group_totals[sgitg] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: SG-0 scans the per-SG totals into per-SG bases
        if (sgitg == 0) {
            int val = (tiisg < (ushort)sgcpr) ? group_totals[tiisg] : 0;
            int prefix = simd_prefix_exclusive_sum(val);
            int total = simd_sum(val);
            if (tiisg < (ushort)sgcpr) {
                group_bases[tiisg] = prefix;
            }
            if (tiisg == 0) {
                tg_total_count = total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: per-thread compacted offset within the chunk
        int offset_in_chunk = simd_prefix + group_bases[sgitg];

        // Step 6: scatter (slot 0 reserved for the final count)
        if (flag) {
            dst[1 + running_total + offset_in_chunk] = chunk_start + (int)tiitg;
        }

        // Step 7: advance the running total for the next chunk
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            running_total += tg_total_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 8: write the final count to dst[0], rounded down to a multiple of 16.
    if (tiitg == 0) {
        dst[0] = (running_total / 16) * 16;
    }
}


///////////////////////////////////////////////////////////////////////////////
// 3-mask sparse-indexing kernel — for q/k/v which share the same post-attention-
// norm input x but have different per-projection thresholds (τ_q, τ_k, τ_v).
//
// Single pass over x: each lane reads x[k] once and computes 3 flags.
// Three independent SIMD prefix sums + threadgroup reductions + scatter writes
// produce 3 count-prefix sparse_indices buffers — packed into a single
// concatenated output of size 3 * (K + 1) int32 (caller slices in Python).
//
// Output layout:
//   dst[0 .. K]            = [n_q, idx_q_0, ..., idx_q_{n_q-1}, 0...]
//   dst[K+1 .. 2K+1]       = [n_k, idx_k_0, ..., idx_k_{n_k-1}, 0...]
//   dst[2K+2 .. 3K+2]      = [n_v, idx_v_0, ..., idx_v_{n_v-1}, 0...]
//
// Single TG, 1024 threads, loops over K in chunks of 1024.
///////////////////////////////////////////////////////////////////////////////

template <typename T>
[[kernel]] void zigzag_sparse_indexing_qkv_v2_impl(
    device const T*       x          [[buffer(0)]],
    device int*           dst        [[buffer(1)]],
    constant const int&   K          [[buffer(2)]],
    constant const float& tau_q      [[buffer(3)]],
    constant const float& tau_k      [[buffer(4)]],
    constant const float& tau_v      [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]) {

    // PARALLEL DESIGN: 3 TGs dispatched (tgpig.x ∈ {0, 1, 2}). Each TG works
    // independently on the SAME `x` (cache-shared across TGs) but with its
    // assigned threshold, producing its slot of `dst`. The Metal scheduler
    // runs all 3 TGs concurrently on M-series GPUs (M2 Ultra has ≫ 3
    // available threadgroup slots). Real parallelism, not sequential ILP
    // inside one TG.
    constexpr int ntg = 1024;
    constexpr int sgcpr = ntg / 32;

    // Pick this TG's mask via tgpig.x.
    float tau;
    int dst_offset;
    if (tgpig.x == 0)      { tau = tau_q; dst_offset = 0; }
    else if (tgpig.x == 1) { tau = tau_k; dst_offset = K + 1; }
    else                    { tau = tau_v; dst_offset = 2 * (K + 1); }

    threadgroup int group_totals[32];
    threadgroup int group_bases[32];
    threadgroup int tg_total_count;
    threadgroup int running_total;

    if (tiitg == 0) {
        running_total = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int num_chunks = (K + ntg - 1) / ntg;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * ntg;
        const int chunk_end = min(chunk_start + ntg, K);
        const int chunk_size = chunk_end - chunk_start;

        // Step 1: load x once for this chunk; build flag for THIS TG's mask.
        int flag = 0;
        if ((int)tiitg < chunk_size) {
            float v = (float)x[chunk_start + (int)tiitg];
            float av = (v < 0.0f) ? -v : v;
            flag = (av > tau) ? 1 : 0;
        }

        // Step 2: SIMD prefix sum.
        int simd_prefix = simd_prefix_exclusive_sum(flag);
        int simd_total = simd_sum(flag);

        // Step 3: per-SG totals → TG memory.
        if (tiisg == 0) {
            group_totals[sgitg] = simd_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: SG 0 scans the per-SG totals.
        if (sgitg == 0) {
            int val = (tiisg < (ushort)sgcpr) ? group_totals[tiisg] : 0;
            int prefix = simd_prefix_exclusive_sum(val);
            int total = simd_sum(val);
            if (tiisg < (ushort)sgcpr) {
                group_bases[tiisg] = prefix;
            }
            if (tiisg == 0) {
                tg_total_count = total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: scatter.
        int off = simd_prefix + group_bases[sgitg];
        if (flag) {
            dst[dst_offset + 1 + running_total + off] = chunk_start + (int)tiitg;
        }

        // Step 6: advance running total for next chunk.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg == 0) {
            running_total += tg_total_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 7: write final count for THIS TG's slot.
    if (tiitg == 0) {
        dst[dst_offset] = (running_total / 16) * 16;
    }
}


///////////////////////////////////////////////////////////////////////////////
// M-skip post-pass: zeroes output rows whose M-position is NOT in
// `active_indices` (count-prefix int32, [n, idx_0, ..., idx_{n-1}], sorted).
//
// First-cut implementation: each thread checks one output position via binary
// search through active_indices. A later optimization will fuse the row-skip
// into the dense GEMV itself so the inactive rows are never computed at all.
///////////////////////////////////////////////////////////////////////////////

[[kernel]] void zigzag_mskip_zero_inactive(
    device float* y                       [[buffer(0)]],
    device const int* active_indices      [[buffer(1)]],
    constant const int& M                 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= (uint)M) return;
  int target = (int)tid;
  int n = active_indices[0];
  int lo = 1;
  int hi = 1 + n;
  bool found = false;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int val = active_indices[mid];
    if (val < target) {
      lo = mid + 1;
    } else if (val > target) {
      hi = mid;
    } else {
      found = true;
      break;
    }
  }
  if (!found) {
    y[tid] = 0.0f;
  }
}


///////////////////////////////////////////////////////////////////////////////
// fused_silu_mskip_qmv_impl — fused FFN block kernel for DejaVu-up.
//
// Inputs (per TG):
//   gate_out : (M_inter,) — pre-computed dense gate_proj output
//   x        : (K,)       — input to up_proj (post-norm activation)
//   w/sc/bi  : up_proj's quantized weights, M-major layout (mlx native)
//   threshold: τ for the silu mask
//
// Output:
//   intermediate : (M_inter,) where
//                  intermediate[m] = silu(gate_out[m]) * up_proj[m] @ x
//                  if |silu(gate_out[m])| > τ, else 0.
//
// This fuses 4 ops into one dispatch:
//   1. silu(gate_out)         (was elementwise dispatch)
//   2. |silu| > τ mask build  (was abs + threshold dispatch)
//   3. mskip up_proj          (was mskip_qmv dispatch)
//   4. silu * up multiply     (was elementwise multiply dispatch)
//
// Design (mirrors mskip_qmv_impl): each simdgroup handles 1 row. Lane 0
// computes silu(gate_out[m]) and decides; broadcasts to other lanes via
// simd_broadcast. Dead rows early-exit before any up_proj weight load.
//
// Template-friendly: T is the activation dtype (half or bfloat16_t).
///////////////////////////////////////////////////////////////////////////////

template <typename T, int group_size, int bits, int num_simdgroups>
[[kernel]] void fused_silu_mskip_qmv_impl(
    const device uint32_t*    w           [[buffer(0)]],   // up_proj weight
    const device T*           scales      [[buffer(1)]],
    const device T*           biases      [[buffer(2)]],
    const device T*           x           [[buffer(3)]],   // post-norm x for up_proj
    const device T*           gate_out    [[buffer(4)]],   // dense gate_proj output
    device T*                 y           [[buffer(5)]],   // intermediate = silu(gate)*up (0 at masked)
    constant const int&       in_vec_size [[buffer(6)]],   // K
    constant const int&       out_vec_size [[buffer(7)]],  // M_inter
    constant const float&     threshold   [[buffer(8)]],
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]]) {

  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const int out_row = tid.y * num_simdgroups + simd_gid;
  if (out_row >= out_vec_size) return;

  // Step 1: silu(gate_out[m]) + threshold check. Only lane 0 computes;
  // broadcast to all lanes via simd_broadcast.
  float silu_val;
  bool active;
  if (simd_lid == 0) {
    float g = (float)gate_out[out_row];
    float sg = 1.0f / (1.0f + metal::precise::exp(-g));   // sigmoid
    silu_val = g * sg;                                     // silu(g) = g * sigmoid(g)
    active = (silu_val > threshold) || (silu_val < -threshold);
  }
  silu_val = simd_broadcast(silu_val, 0);
  active = simd_broadcast((uint)active, 0) != 0;

  // Early-exit on dead row: write exact 0 to intermediate.
  if (!active) {
    if (simd_lid == 0) {
      y[out_row] = static_cast<T>(0.0f);
    }
    return;
  }

  // Step 2: standard mskip-style M-skip GEMV row-walk (active row only).
  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    U s = scales[0];
    U b = biases[0];
    result += qdot<U, values_per_thread, bits>(ws, x_thread, s, b, sum);
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  result = simd_sum(result);
  if (simd_lid == 0) {
    // Step 3: multiply by silu_val and write.
    y[out_row] = static_cast<T>(result * silu_val);
  }
}


///////////////////////////////////////////////////////////////////////////////
// mskip_qmv_impl — mask-aware M-skip GEMV on mlx's standard M-major
// quantized weight layout (NOT zigzag). Designed for DejaVu-up FFN
// projections (gate_proj, up_proj) where the M-axis sparsity is determined
// by a per-row mask, so dead rows can be skipped entirely.
//
// Each simdgroup handles ONE output row (results_per_simdgroup = 1). If
// that row's mask is 0 the SG writes 0 and returns before any weight load —
// real memory-bandwidth savings proportional to the dead fraction.
//
// Adapted from mlx's `qmv_fast_impl` (quantized.h:750). Same load_vector +
// qdot helpers; only difference is the per-row mask check and the single
// row per SG (vs 4 rows in qmv_fast).
///////////////////////////////////////////////////////////////////////////////

template <typename T, int group_size, int bits, int num_simdgroups>
[[kernel]] void mskip_qmv_impl(
    const device uint32_t*    w           [[buffer(0)]],
    const device T*           scales      [[buffer(1)]],
    const device T*           biases      [[buffer(2)]],
    const device T*           x           [[buffer(3)]],
    const device uchar*       mask        [[buffer(4)]],
    device T*                 y           [[buffer(5)]],
    constant const int&       in_vec_size [[buffer(6)]],
    constant const int&       out_vec_size [[buffer(7)]],
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]]) {

  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result = 0;

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * num_simdgroups + simd_gid;

  if (out_row >= out_vec_size) return;

  // Mask early-exit: dead row writes 0 and returns before any weight load.
  if (mask[out_row] == 0) {
    if (simd_lid == 0) {
      y[out_row] = static_cast<T>(0.0f);
    }
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
    U s = scales[0];
    U b = biases[0];
    result += qdot<U, values_per_thread, bits>(ws, x_thread, s, b, sum);
    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  result = simd_sum(result);
  if (simd_lid == 0) {
    y[out_row] = static_cast<T>(result);
  }
}


///////////////////////////////////////////////////////////////////////////////
// Bits-aware unpack helper: reads `4 * bits / 8` bytes from `w_byte_ptr`,
// unpacks 4 dequantized integer values, multiplies each by `x_scale_cur`,
// and accumulates into `acc[0..3]`. Caller is responsible for `bias_total`,
// which is summed once at the end (independent of the unpack loop).
//
// Per-iteration byte stride:
//   bits=4 → 2 bytes (4 nibbles in a uint16_t)
//   bits=6 → 3 bytes (4 × 6-bit values packed across 3 bytes)
//   bits=8 → 4 bytes (4 int8 values in a uint32_t)
//
// `if (bits == N)` branches are compile-time selected since `bits` is a
// template constant — MSL's optimizer drops the inactive arms.
///////////////////////////////////////////////////////////////////////////////

template <int bits>
inline void zigzag_unpack4_and_accumulate(
    const device uint8_t* w_byte_ptr,
    float x_scale_cur,
    thread float* acc) {
    if (bits == 4) {
        uint16_t word = ((const device uint16_t*)w_byte_ptr)[0];
        acc[0] += x_scale_cur * (word & 0x000f);
        acc[1] += x_scale_cur * ((word & 0x00f0) >> 4);
        acc[2] += x_scale_cur * ((word & 0x0f00) >> 8);
        acc[3] += x_scale_cur * ((word & 0xf000) >> 12);
    } else if (bits == 8) {
        // 4 int8 values densely packed into one 32-bit word, little-endian byte order.
        uint32_t word = ((const device uint32_t*)w_byte_ptr)[0];
        acc[0] += x_scale_cur * (word & 0x000000ffu);
        acc[1] += x_scale_cur * ((word & 0x0000ff00u) >> 8);
        acc[2] += x_scale_cur * ((word & 0x00ff0000u) >> 16);
        acc[3] += x_scale_cur * ((word & 0xff000000u) >> 24);
    } else if (bits == 6) {
        // 4 × 6-bit values densely packed into 3 bytes (24 bits).
        // Layout matches mlx core's affine_dequantize for bits=6 (see quantized.h:2575).
        uint32_t b0 = w_byte_ptr[0];
        uint32_t b1 = w_byte_ptr[1];
        uint32_t b2 = w_byte_ptr[2];
        acc[0] += x_scale_cur * (b0 & 0x3f);
        acc[1] += x_scale_cur * (((b0 >> 6) & 0x03) + ((b1 & 0x0f) << 2));
        acc[2] += x_scale_cur * (((b1 >> 4) & 0x0f) + ((b2 & 0x03) << 4));
        acc[3] += x_scale_cur * ((b2 >> 2) & 0x3f);
    }
}


///////////////////////////////////////////////////////////////////////////////
// zigzag_qmv_fast_impl — optimized variant for the overhead-bound regime
// (≤4 MB weights) where the original threadgroup-atomic-add design loses to
// `mx.quantized_matmul`. Design:
//
//   - One threadgroup per output row-band (no cross-TG aggregation, no atomics).
//   - Each simdgroup handles `results_per_simdgroup` consecutive rows.
//   - 32 lanes per simdgroup split the K-walk; each lane accumulates partial
//     per-row sums in registers, then `simd_sum` combines them.
//   - Lane 0 writes the final results directly to global y (no zero-fill prepass).
//
// Tuning constants are template parameters (results_per_simdgroup * num_simdgroups
// must equal group_size so that one TG covers one band exactly).
///////////////////////////////////////////////////////////////////////////////

template <typename T, int group_size, int bits, int num_simdgroups, int results_per_simdgroup>
[[kernel]] void zigzag_qmv_fast_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device float*               out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {

    static_assert(num_simdgroups * results_per_simdgroup == group_size,
                  "num_simdgroups * results_per_simdgroup must equal group_size");
    static_assert(results_per_simdgroup % 4 == 0,
                  "results_per_simdgroup must be a multiple of 4 (unpack4 helper)");

    const int band = tid.x;
    const int row_in_band = simd_gid * results_per_simdgroup;   // first row this SG owns
    const int row_start = band * group_size + row_in_band;

    // Per-K weight chunk: group_size * bits / 8 bytes per (band, k).
    const int k_stride_bytes = group_size * bits / 8;
    // Offset (in bytes) within a (band, k) pack to the first byte for this SG's rows.
    const int sg_byte_offset_in_pack = row_in_band * bits / 8;
    // Pointer to (band, k=0, row_in_band) byte.
    const device uint8_t* ws_base = (const device uint8_t*)w_zz
        + band * K * k_stride_bytes + sg_byte_offset_in_pack;

    // Each lane handles K / 32 K-positions. simd_lid steps through K.
    float acc[results_per_simdgroup] = {0.0f};
    float bias_total = 0.0f;

    constexpr int per_unpack_bytes = 4 * bits / 8;

    // simd_lid = lane index ∈ [0, 32). Each lane walks K-positions in stride of 32.
    for (int k = (int)simd_lid; k < K; k += 32) {
        // Scale/bias for this (band, k): stride 1 along k, band-major.
        float x_val = (float)x[k];
        float scale = (float)scales_zz[band * K + k];
        float bias = (float)biases_zz[band * K + k];
        float x_scale = x_val * scale;
        bias_total += x_val * bias;

        // Load weights for this lane's K-position, all `results_per_simdgroup` rows.
        const device uint8_t* w_ptr = ws_base + k * k_stride_bytes;
        #pragma unroll
        for (int g = 0; g < results_per_simdgroup / 4; g++) {
            zigzag_unpack4_and_accumulate<bits>(w_ptr + g * per_unpack_bytes, x_scale, &acc[g * 4]);
        }
    }

    // Reduce across lanes (32 lanes contribute to the same `results_per_simdgroup` rows).
    float bias_total_simdsum = simd_sum(bias_total);
    #pragma unroll
    for (int r = 0; r < results_per_simdgroup; r++) {
        float row_sum = simd_sum(acc[r]);
        if (simd_lid == 0) {
            out[row_start + r] = row_sum + bias_total_simdsum;
        }
    }
}


template <typename T, int group_size, int bits, int num_simdgroups, int threadgroups_per_band>
[[kernel]] void zigzag_qmv_dense_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {

    const uint thread_id = simd_gid * 32 + simd_lid;  
    
    constexpr int packs_per_thread = 4; //bits == 2 ? 1 : 2;
    //constexpr int num_simdgroups = num_simdgroups; 
    constexpr int results_per_simdgroup = group_size; // each simdgroup will only work on 1 band of group_size rows final results.
    constexpr int pack_factor = get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread; // number of values (elements of x) processed by each thread.
    constexpr int block_size = values_per_thread * SIMD_SIZE; // number of values (elements of x) processed by each simdgroup in total.
    constexpr int scale_step_per_thread = group_size / values_per_thread; // scale_step_per_thread also equals to number of threads that share the same scale and bias, since threads in the same simdgroup work on the same group of rows.

    const device uint8_t* ws = (const device uint8_t*)w_zz; // treat weights as bytes for easier indexing with bits.

    typedef float U;

    // prepare atomic shared memory for accumulation within threadgroup. each threadgroup will compute results for a band of group_size rows.
    constexpr float SCALE_FACTOR = 65536.0f; // float to int
    threadgroup atomic<int> shared_out[group_size]; 

    // initialize shared_out to 0.0f
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        atomic_store_explicit(&shared_out[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure shared_out is initialized before

    // local result accumulator in registers for each thread.
    float acc[values_per_thread] = {0.0f}; // each thread accumulates values_per_thread output elements. 
    float bias_total = 0.0f; // accumulate total bias for the thread, since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

    // compute the starting row index for this threadgroup
    int band_id = tid.x; // each threadgroup works on one band of rows, and each band has group_size rows.
    int row_start = band_id * group_size;
    
    // compute the starting group index for this simdgroup. K groups in a row, divided by threadgroups_per_band * num_simdgroups simdgroups.
    int groups_per_simdgroup = K / (threadgroups_per_band * num_simdgroups);
    int groups_start_id = groups_per_simdgroup * (simd_gid + tid.y * num_simdgroups);
    int groups_start_id_thread = groups_start_id + (simd_lid * values_per_thread / group_size);

    // for each thread, get the x offset, basically the group index the thread will start from.
    int x_offset = groups_start_id + simd_lid * values_per_thread / group_size;
    // for each thread, get the w offset, ws uint8. offset = element count * element bitwidth / 8
    int w_offset = groups_start_id_thread * group_size * bits / 8 + row_start * K * bits / 8 + ((simd_lid * values_per_thread % group_size) * bits / 8);

    x += x_offset; // move the x pointer to the starting offset for this thread.
    ws += w_offset; // move the weight pointer to the starting offset for this thread.
    scales_zz += groups_start_id_thread + K * band_id; 
    biases_zz += groups_start_id_thread + K * band_id; 

    // calculate the total outer loop step size, essentially total element count processed by each simdgroup, divided by each round of loop processing element count (simdgroup_size * values_per_thread).
    int total_step = (groups_per_simdgroup * group_size + values_per_thread * 32 - 1) / (values_per_thread * 32);
    // outer loop, each simdgroup processes a block of elements from x and corresponding weights, scales, biases, and accumulates partial results to shared_out.
    for (int k = 0; k < total_step; k++) {
        // load x values for the thread. single value with the offset
        float x_cur = x[0];
        float x_scale_cur = x_cur * scales_zz[0]; // apply scale to x. since quantized weight will be multiplied with x, we can apply the scale to x directly.
        float x_bias_cur = x_cur * biases_zz[0]; // apply bias to x. this is an approximation to adding bias after multiplication, but it allows us to reuse the same accumulation for different groups of rows that share the same scale and bias, which is necessary for efficiency.
        bias_total += x_bias_cur; // accumulate bias for the thread. since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

        // load and unpack weights; bits-aware helper handles the per-bits arithmetic.
        // Each helper call consumes (4 * bits / 8) bytes and produces 4 dequantized values.
        constexpr int per_unpack_bytes = 4 * bits / 8;
        for (int i = 0; i < values_per_thread / 4; i++) {
            zigzag_unpack4_and_accumulate<bits>(ws + i * per_unpack_bytes, x_scale_cur, &acc[i * 4]);
        }
        // move pointers for the next loop iteration.
        x += values_per_thread * 32 / group_size;
        ws += values_per_thread * 32 * bits / 8;
        scales_zz += values_per_thread * 32 / group_size;
        biases_zz += values_per_thread * 32 / group_size;
    }

    // apply bias_total to all rows once at the end.
    #pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
        acc[i] += bias_total;
    }

    // write the accumulated results to shared memory atomically.
    for (int i = 0; i < values_per_thread; i++) {
        int idx = simd_lid * values_per_thread % group_size + i;
        atomic_fetch_add_explicit(&shared_out[idx], (int)(acc[i] * SCALE_FACTOR), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure all threads have written their results to shared memory

    // // all the threads write their results to global memory atomically.
    // for (int i = 0; i < values_per_thread; i++) {
    //     int out_index = row_start + simd_lid * values_per_thread % group_size + i;
    //     atomic_fetch_add_explicit(&out[out_index], acc[i], memory_order_relaxed);
    // }

    // write the shared memory results to global memory in parallel
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        float v = atomic_load_explicit(&shared_out[i], memory_order_relaxed) / SCALE_FACTOR; // convert back to float
        atomic_fetch_add_explicit(&out[row_start + i], v, memory_order_relaxed); // write to global memory atomically, since different simdgroups may work on the same group of rows and cause write conflicts.
    }
}

template <typename T, int group_size, int bits, int num_simdgroups, int threadgroups_per_band>
[[kernel]] void zigzag_qmv_sparse_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device const int*             sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {

    const uint thread_id = simd_gid * 32 + simd_lid;  
    
    constexpr int packs_per_thread = 4; //bits == 2 ? 1 : 2;
    //constexpr int num_simdgroups = num_simdgroups; 
    constexpr int results_per_simdgroup = group_size; // each simdgroup will only work on 1 band of group_size rows final results.
    constexpr int pack_factor = get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread; // number of values (elements of x) processed by each thread.
    constexpr int block_size = values_per_thread * SIMD_SIZE; // number of values (elements of x) processed by each simdgroup in total.
    constexpr int scale_step_per_thread = group_size / values_per_thread; // scale_step_per_thread also equals to number of threads that share the same scale and bias, since threads in the same simdgroup work on the same group of rows.

    const device uint8_t* ws = (const device uint8_t*)w_zz; // treat weights as bytes for easier indexing with bits.
    const device int* sparse_indices_cur = sparse_indices + 1; // the first element in sparse_indices is the total number of non-zero elements, so we need to offset by 1 to get the actual indices.
    typedef float U;

    // prepare atomic shared memory for accumulation within threadgroup. each threadgroup will compute results for a band of group_size rows.
    constexpr float SCALE_FACTOR = 65536.0f; // float to int
    threadgroup atomic<int> shared_out[group_size]; 

    // initialize shared_out to 0.0f
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        atomic_store_explicit(&shared_out[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure shared_out is initialized before

    // local result accumulator in registers for each thread.
    float acc[values_per_thread] = {0.0f}; // each thread accumulates values_per_thread output elements. 
    float bias_total = 0.0f; // accumulate total bias for the thread, since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

    // compute the starting row index for this threadgroup
    int band_id = tid.x; // each threadgroup works on one band of rows, and each band has group_size rows.
    int row_start = band_id * group_size;
    
    // compute the starting group index for this simdgroup. K groups in a row, divided by threadgroups_per_band * num_simdgroups simdgroups.
    int n_total = sparse_indices[0]; // total number of non-zero elements in the sparse vector, stored in the first element of sparse_indices.
    int groups_per_simdgroup = n_total / (threadgroups_per_band * num_simdgroups);
    int groups_start_id = groups_per_simdgroup * (simd_gid + tid.y * num_simdgroups);
    int groups_start_count_thread = groups_start_id + (simd_lid * values_per_thread / group_size);
    int groups_end_count_thread = groups_start_id + groups_per_simdgroup;
    int group_cur_count_thread = groups_start_count_thread;
    // // for each thread, get the x offset, basically the group index the thread will start from.
    // int x_offset = groups_start_id + simd_lid * values_per_thread / group_size; // group count
    // // for each thread, get the w offset, ws uint8. offset = element count * element bitwidth / 8
    // int w_offset = sparse_indices_cur[groups_start_id_thread] * group_size * bits / 8 + row_start * K * bits / 8 + ((simd_lid * values_per_thread % group_size) * bits / 8);

    // x += x_offset; // move the x pointer to the starting offset for this thread.
    // ws += w_offset; // move the weight pointer to the starting offset for this thread.
    // scales_zz += sparse_indices_cur[groups_start_id_thread] + K * band_id; 
    // biases_zz += sparse_indices_cur[groups_start_id_thread] + K * band_id; 



    // calculate the total outer loop step size, essentially total element count processed by each simdgroup, divided by each round of loop processing element count (simdgroup_size * values_per_thread).
    int total_step = (groups_per_simdgroup * group_size + values_per_thread * 32 - 1) / (values_per_thread * 32);
    // outer loop, each simdgroup processes a block of elements from x and corresponding weights, scales, biases, and accumulates partial results to shared_out.
    for (int k = 0; k < total_step; k++) {
        int group_cur_id_thread = sparse_indices_cur[group_cur_count_thread]; // get the actual group index for this thread to work on. since the non-zero elements in x are stored in a sparse format with an index array, we need to get the group index from the sparse_indices array using the thread's group count index.
        if (group_cur_count_thread < groups_end_count_thread) { 
            // load x values for the thread. single value with the offset
            float x_cur = x[group_cur_id_thread];
            int scale_offset = group_cur_id_thread + K * band_id; // calculate the offset for scales and biases. since scales and biases are stored in a dense format, we can directly calculate the offset using the actual group index.
            float x_scale_cur = x_cur * scales_zz[scale_offset]; // apply scale to x. since quantized weight will be multiplied with x, we can apply the scale to x directly.
            float x_bias_cur = x_cur * biases_zz[scale_offset]; // apply bias to x. this is an approximation to adding bias after multiplication, but it allows us to reuse the same accumulation for different groups of rows that share the same scale and bias, which is necessary for efficiency.
            bias_total += x_bias_cur; // accumulate bias for the thread. since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

            // load and unpack weights; bits-aware helper handles the per-bits arithmetic.
            const device uint8_t* w_cur_base = ws + row_start * K * bits / 8 + group_cur_id_thread * group_size * bits / 8 + ((simd_lid * values_per_thread % group_size) * bits / 8);
            constexpr int per_unpack_bytes = 4 * bits / 8;
            for (int i = 0; i < values_per_thread / 4; i++) {
                zigzag_unpack4_and_accumulate<bits>(w_cur_base + i * per_unpack_bytes, x_scale_cur, &acc[i * 4]);
            }
        }
        // move pointers for the next loop iteration. 
        group_cur_count_thread += values_per_thread * 32 / group_size; 
        // x += values_per_thread * 32 / group_size; 
        // ws += values_per_thread * 32 * bits / 8;
        // scales_zz += values_per_thread * 32 / group_size;
        // biases_zz += values_per_thread * 32 / group_size;
    }

    // apply bias_total to all rows once at the end.
    #pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
        acc[i] += bias_total;
    }

    // write the accumulated results to shared memory atomically.
    for (int i = 0; i < values_per_thread; i++) {
        int idx = simd_lid * values_per_thread % group_size + i;
        atomic_fetch_add_explicit(&shared_out[idx], (int)(acc[i] * SCALE_FACTOR), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure all threads have written their results to shared memory

    // // all the threads write their results to global memory atomically. 
    // for (int i = 0; i < values_per_thread; i++) {
    //     int out_index = row_start + simd_lid * values_per_thread % group_size + i;
    //     atomic_fetch_add_explicit(&out[out_index], acc[i], memory_order_relaxed);
    // }

    // write the shared memory results to global memory in parallel
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        float v = atomic_load_explicit(&shared_out[i], memory_order_relaxed) / SCALE_FACTOR; // convert back to float
        atomic_fetch_add_explicit(&out[row_start + i], v, memory_order_relaxed); // write to global memory atomically, since different simdgroups may work on the same group of rows and cause write conflicts.
    }
}
