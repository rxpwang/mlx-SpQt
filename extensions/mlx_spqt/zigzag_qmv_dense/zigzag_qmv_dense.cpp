// Copyright © 2023-2025 Apple Inc.

#include <dlfcn.h>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "zigzag_qmv_dense/zigzag_qmv_dense.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/backend/gpu/copy.h"                                             
#endif

namespace spqt_ext {

namespace {

// A helper function to find the location of the current binary on disk.
// The Metal library ("mlx_spqt.metallib") should be in the same directory.
std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

/**
    *  zigzag_qmv_dense: perform a zigzag-ordered quantized matrix-vector multiplication
    *  with a dense matrix.
    *   *  Args:
    *      x (array): activations, shape (B, K), dtype fp16.
    *      w_zz (array): zigzag-packed weights, shape (M/gs, K/gs, gs*bits/32), dtype uint32.
    *      scales (array): zigzag scales, shape (M/gs, K), dtype fp16.
    *      biases (array): zigzag biases, shape (M/gs, K), dtype fp16.
    *      group_size (int): size of zigzag groups (gs).
    *      bits (int): number of bits per quantized weight.
    *   *  Returns: 
    *      array: result of the matrix-vector multiplication as float32, shape (B, M).
 **/
mx::array zigzag_qmv_dense(
    const mx::array& x,
    const mx::array& w_zz,
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int bits,
    mx::StreamOrDevice s
) {

  const int n_bands = scales.shape(0);
  const int M = n_bands * group_size;
  const int B = x.shape(0);
  mx::Shape out_shape{B, M};
  
  // Construct the array as the output of the ZigzagQmvDense primitive
  // with the input array as input
  return mx::array(
      /* const mx::Shape& shape = */ out_shape,
      /* mx::Dtype dtype = */ mx::float32,
      /* std::shared_ptr<mx::Primitive> primitive = */
      std::make_shared<ZigzagQmvDense>(to_stream(s), group_size, bits),
      /* const std::vector<mx::array>& inputs = */ {w_zz, x, scales, biases});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////


void ZigzagQmvDense::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
    return;
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

/** Evaluate primitive on GPU */
void ZigzagQmvDense::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
    
    // prepare input and output
    auto& w_zz = inputs[0];
    auto& x = inputs[1];
    auto& scales = inputs[2];
    auto& biases = inputs[3];
    auto& out = outputs[0];

    // Get the Metal device and command queue
    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    // allocate output buffer on the device
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Load the Metal library and get the kernel function
    auto lib = d.get_library("mlx_spqt", current_binary_dir());
    auto kernel = d.get_kernel("zigzag_qmv_dense_half_gs64_b4_NSG4_TG4", lib);

    // zero-fill output buffer (since the kernel will be doing atomic adds)
    {
      auto zero_kernel = d.get_kernel("zigzag_zero_fill_float", lib);
      auto& zfe = mx::metal::get_command_encoder(s);
      zfe.set_compute_pipeline_state(zero_kernel);
      zfe.set_output_array(out, 0);
      const size_t n = out.size();
      const size_t tgp = std::min(n, zero_kernel->maxTotalThreadsPerThreadgroup());
      zfe.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(tgp, 1, 1));
    }

    // set up the encoder + bind buffers
    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(w_zz, 0);
    compute_encoder.set_input_array(x, 1);
    compute_encoder.set_input_array(scales, 2);
    compute_encoder.set_input_array(biases, 3);
    compute_encoder.set_output_array(out, 4);
    const int K = x.shape(-1);
    compute_encoder.set_bytes(K, 5);

    constexpr int num_simdgroups = 4;
    constexpr int threadgroups_per_band = 4;
    constexpr int simdgroup_size = 32;
    const int M = out.shape(-1);
 
    // dispatch threads
    MTL::Size group_dims = MTL::Size(num_simdgroups * simdgroup_size, 1, 1);
    MTL::Size grid_dims = MTL::Size(M / group_size_, threadgroups_per_band, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

#else // Metal is not available

/** Fail evaluation on GPU */
void ZigzagQmvDense::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& out) {
  throw std::runtime_error("ZigzagQmvDense has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<mx::array> ZigzagQmvDense::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("ZigzagQmvDense has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<mx::array> ZigzagQmvDense::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  throw std::runtime_error("ZigzagQmvDense has no vjp implementation.");
}

/** Vectorize primitive along given axis */
std::pair<std::vector<mx::array>, std::vector<int>> ZigzagQmvDense::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("ZigzagQmvDense has no vmap implementation.");
}

/** Equivalence check **/
bool ZigzagQmvDense::is_equivalent(const Primitive& other) const {
  const ZigzagQmvDense& r_other = static_cast<const ZigzagQmvDense&>(other);
  return group_size_ == r_other.group_size_ && bits_ == r_other.bits_;
}

} // namespace spqt_ext
