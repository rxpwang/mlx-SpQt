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
    *  Smoke test:  test the atomicity of updates to the same output location by launching multiple threads that update the same output buffer location.
    *   *  Args:
    *       N (int): number of output elements.
    *   *  Returns:
    *       array: output with first element updated by multiple threads, shape (N, ).
 **/
mx::array smoke_atomic(
    int N, // Number of output elements
    mx::StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class SmokeAtomic : public mx::Primitive {
 public:
  explicit SmokeAtomic(mx::Stream stream, int N)
      : mx::Primitive(stream), N_(N) {};

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
    return "SmokeAtomic";
  }

  /** Equivalence check **/
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int N_;
};



} // namespace spqt_ext
