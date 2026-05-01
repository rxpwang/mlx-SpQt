// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

[[kernel]] void smoke_threadgroup(
    device float* out [[buffer(0)]],
    constant const int& N [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint num_sgs [[simdgroups_per_threadgroup]]) {
  threadgroup float shared[32]; // Assuming a maximum of 32 threads in a threadgroup for simplicity.
  if (tid >= N) return; // Only one thread computes the atomic update for simplicity.
  float v = 1.0f; // Each thread contributes 1.0f to the sum.
  v = simd_sum(v); // Sum across the threadgroup.
  if (sg_lane == 0) { // Let the first thread in each SIMD group write the partial sum to shared memory.
    shared[sg_id] = v;
  }
  if (sg_id == 0) { 
    threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all partial sums are written to shared memory.
    float tmp_v = shared[sg_lane]; // Let the first SIMD group read the partial sums.
    tmp_v = simd_sum(tmp_v); // Sum the partial sums to get the final result.
    if (sg_lane == 0) { // Let the first thread write the final result to output.
      out[0] = tmp_v;
    }
  }
}
