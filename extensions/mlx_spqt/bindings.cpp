// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "axpby/axpby.h"
#include "smoke_uint32/smoke_uint32.h"
#include "smoke_qdot/smoke_qdot.h"
#include "smoke_atomic/smoke_atomic.h"
#include "smoke_threadgroup/smoke_threadgroup.h"
#include "zigzag_qmv_dense/zigzag_qmv_dense.h"
#include "zigzag_qmv_sparse/zigzag_qmv_sparse.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_spqt_ext, m) {
  m.doc() = "Sample extension for MLX";

  m.def(
      "axpby",
      &spqt_ext::axpby,
      "x"_a,
      "y"_a,
      "alpha"_a,
      "beta"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");
  
  m.def(
      "smoke_uint32",
      &spqt_ext::smoke_uint32,
      "w_packed"_a,
      "N"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Smoke test: extract nibbles from a uint32-packed buffer.

        Args:
            w_packed (array): uint32-packed input.
            N (int): number of output elements.
        
        Returns:
            array: nibble values as float32, shape (N, ).
      )");
  
  m.def(
      "smoke_qdot",
      &spqt_ext::smoke_qdot,
      "w_packed"_a,
      "x"_a,
      "N"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Smoke test: extract nibbles from a uint32-packed buffer and compute dot product with x.

        Args:
            w_packed (array): uint32-packed input.
            x (array): Input array.
            N (int): number of output elements.
        
        Returns:
            array: nibble values as float32, shape (N, ).
      )");
  
  m.def(
      "smoke_atomic",
      &spqt_ext::smoke_atomic,
      "N"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Smoke test: test the atomicity of updates to the same output location by launching multiple threads that update the same output buffer location.

        Args:
            N (int): number of output elements.

        Returns:
            array: output with first element updated by multiple threads, shape (N, ).
      )");
  m.def(
      "smoke_threadgroup",
      &spqt_ext::smoke_threadgroup,
      "N"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Smoke test: test within one threadgroup of M threads, reduce values across simdgroups using simd_sum, threadgroup memory, threadgroup_barrier, and a second simd_sum to ensure all threads see the same final result.

        Args:
            N (int): number of output elements.

        Returns:
            array: output with first element updated by multiple threads, shape (N, ).
      )");

  m.def(
        "zigzag_qmv_dense",
        &spqt_ext::zigzag_qmv_dense,
        "x"_a,
        "w_zz"_a,
        "scales"_a,
        "biases"_a,
        "group_size"_a = 64,
        "bits"_a = 4,
        nb::kw_only(),
        "stream"_a = nb::none(),
        R"(
            Perform a zigzag-ordered quantized matrix-vector multiplication with a dense matrix.
    
            Args:
                x (array): activations, shape (B, K), dtype fp16.
                w_zz (array): zigzag-packed weights, shape (M/gs, K/gs, gs*bits/32), dtype uint32.
                scales (array): zigzag scales, shape (M/gs, K), dtype fp16.
                biases (array): zigzag biases, shape (M/gs, K), dtype fp16.
                group_size (int): size of zigzag groups (gs).
                bits (int): number of bits per quantized weight.
    
            Returns:
                array: result of the matrix-vector multiplication as float32, shape (B, M).
        )");

    m.def(
        "zigzag_qmv_sparse",
        &spqt_ext::zigzag_qmv_sparse,
        "x"_a,
        "sparse_indices"_a,
        "w_zz"_a,
        "scales"_a,
        "biases"_a,
        "group_size"_a = 64,
        "bits"_a = 4,
        nb::kw_only(),
        "stream"_a = nb::none(),
        R"(
            Perform a zigzag-ordered quantized matrix-vector multiplication with a matrix and a sparse vector with a index array indicating the non-zero positions.
    
            Args:
                x (array): activations, shape (B, K), dtype fp16.
                sparse_indices (array): indices of non-zero elements in the sparse vector, shape (B, K), dtype int32.
                w_zz (array): zigzag-packed weights, shape (M/gs, K/gs, gs*bits/32), dtype uint32.
                scales (array): zigzag scales, shape (M/gs, K), dtype fp16.
                biases (array): zigzag biases, shape (M/gs, K), dtype fp16.
                group_size (int): size of zigzag groups (gs).
                bits (int): number of bits per quantized weight.
    
            Returns:
                array: result of the matrix-vector multiplication as float32, shape (B, M).
        )");
}
