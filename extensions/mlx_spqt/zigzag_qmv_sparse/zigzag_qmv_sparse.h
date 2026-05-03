// Copyright © 2023-2025 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace spqt_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////
/**
    *  zigzag_qmv_sparse: perform a zigzag-ordered quantized matrix-vector multiplication
    *  with a matrix and a sparse vector and a index array indicating the non-zero positions.
    *   *  Args:
    *      x (array): activations, shape (B, K), dtype fp16.
    *      sparse_indices (array): indices of non-zero elements in the sparse vector, shape (B, K), dtype int32.
    *      w_zz (array): zigzag-packed weights, shape (M/gs, K, gs*bits/32), dtype uint32.
    *      scales (array): zigzag scales, shape (M/gs, K), dtype fp16.
    *      biases (array): zigzag biases, shape (M/gs, K), dtype fp16.
    *      group_size (int): size of zigzag groups (gs).
    *      bits (int): number of bits per quantized weight.
    *   *  Returns: 
    *      array: result of the matrix-vector multiplication as float32, shape (B, M).
 **/
mx::array zigzag_qmv_sparse(
    const mx::array& x,
    const mx::array& sparse_indices,
    const mx::array& w_zz,
    const mx::array& scales,
    const mx::array& biases,
    int group_size = 64,
    int bits = 4,
    int num_simdgroups = 4,
    int threadgroups_per_band = 4,
    mx::StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class ZigzagQmvSparse : public mx::Primitive {
 public:
  explicit ZigzagQmvSparse(mx::Stream stream, int group_size, int bits, int num_simdgroups, int threadgroups_per_band)
      : mx::Primitive(stream), group_size_(group_size), bits_(bits), num_simdgroups_(num_simdgroups), threadgroups_per_band_(threadgroups_per_band) {};

  /**
   * A primitive must know how to evaluate itself on the CPU/GPU
   * for the given inputs and populate the output array.
   *
   * To avoid unnecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  /** The Jacobian-vector product. */
  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  /** The name of primitive. */
  const char* name() const override {
    return "ZigzagQmvSparse";
  }

  /** Equivalence check **/
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int group_size_;
  int bits_;
  int num_simdgroups_;
  int threadgroups_per_band_;
};



} // namespace spqt_ext
