// Copyright © 2023-2025 Apple Inc.

#include <dlfcn.h>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "smoke_uint32/smoke_uint32.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
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
    *  Smoke test: extract nibbles from a uint32-packed buffer.
    *   *  Args:
    *       w_packed (array): uint32-packed input.
    *       N (int): number of output elements.
    *   *  Returns:
    *       array: nibble values as float32, shape (N, ).
 **/
mx::array smoke_uint32(
    const mx::array& w_packed, // Input mx::array w_packed
    int N, // Number of output elements
    mx::StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  
  // Construct the array as the output of the SmokeUint32 primitive
  // with the input array as input
  return mx::array(
      /* const mx::Shape& shape = */ {N},
      /* mx::Dtype dtype = */ mx::float32,
      /* std::shared_ptr<mx::Primitive> primitive = */
      std::make_shared<SmokeUint32>(to_stream(s), N),
      /* const std::vector<mx::array>& inputs = */ {w_packed});
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////


void SmokeUint32::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
    return;
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

/** Evaluate primitive on GPU */
void SmokeUint32::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
    
    // prepare input and output
    auto& w_packed = inputs[0];
    auto& out = outputs[0];

    // Get the Metal device and command queue
    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    // allocate output buffer on the device
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // Load the Metal library and get the kernel function
    auto lib = d.get_library("mlx_spqt", current_binary_dir());
    auto kernel = d.get_kernel("smoke_uint32", lib);

    // set up the encoder + bind buffers
    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(w_packed, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(N_, 2);
 
    // dispatch threads
    size_t tgp_size = std::min((size_t)N_, kernel->maxTotalThreadsPerThreadgroup());
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    MTL::Size grid_dims = MTL::Size(N_, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
}

#else // Metal is not available

/** Fail evaluation on GPU */
void SmokeUint32::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& out) {
  throw std::runtime_error("SmokeUint32 has no GPU implementation.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

/** The Jacobian-vector product. */
std::vector<mx::array> SmokeUint32::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("SmokeUint32 has no jvp implementation.");
}

/** The vector-Jacobian product. */
std::vector<mx::array> SmokeUint32::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>&) {
  throw std::runtime_error("SmokeUint32 has no vjp implementation.");
}

/** Vectorize primitive along given axis */
std::pair<std::vector<mx::array>, std::vector<int>> SmokeUint32::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("SmokeUint32 has no vmap implementation.");
}

/** Equivalence check **/
bool SmokeUint32::is_equivalent(const Primitive& other) const {
  const SmokeUint32& r_other = static_cast<const SmokeUint32&>(other);
  return N_ == r_other.N_;
}

} // namespace spqt_ext
