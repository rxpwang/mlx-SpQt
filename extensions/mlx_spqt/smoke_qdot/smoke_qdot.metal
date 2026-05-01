// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"   
#include "mlx/backend/metal/kernels/quantized_utils.h"    
#include "mlx/backend/metal/kernels/quantized.h"

[[kernel]] void smoke_qdot(
    device const uint32_t* w_packed [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant const int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid > 0) return; // Only one thread computes the dot product for simplicity.
  float x_thread[16]; // Assuming x has 16 elements for the dot product.
  float sum = load_vector<float, float, 16, 4>(x, x_thread); // Load x into registers.

  auto w_bytes = (const device uint8_t*)w_packed; // Treat w_packed as bytes for qdot.

  float result = qdot<float, 16, 4>(w_bytes, x_thread, 1.0f, 0.0f, sum); // Compute dot product of 16 nibbles with x.,

  out[0] = result; // Write the result to output.
}
