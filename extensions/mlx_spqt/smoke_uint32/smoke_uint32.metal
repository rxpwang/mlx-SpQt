// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

[[kernel]] void smoke_uint32(
    device const uint32_t* w_packed [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant const int& N [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= N) return;
  uint word_idx = tid / 8;
  uint nibble_pos = tid % 8;
  uint nibble = (w_packed[word_idx] >> (4 * nibble_pos)) & 0xF;
  out[tid] = float(nibble);
}
