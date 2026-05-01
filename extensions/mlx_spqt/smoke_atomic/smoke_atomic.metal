// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

[[kernel]] void smoke_atomic(
    device atomic_float* out [[buffer(0)]],
    constant const int& N [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= N) return; // Only one thread computes the atomic update for simplicity.
  atomic_fetch_add_explicit(out, 1.0f, memory_order_relaxed); // Perform atomic update.
}
