# I3 — Python bindings + Python helpers

Goal: `mx.zigzag_qmv_dense` and `mx.zigzag_qmv_sparse` are
callable from Python, and `from mlx.spqt import quantize_zigzag,
dequantize_zigzag` works.

After I3, the entire user-facing API is in place. I4 covers tests
and I5 the headline-perf reproduction.

## Files

### Modify

- `python/src/ops.cpp` — add two `m.def(...)` entries near
  `quantized_matmul` at line 4311.

### Create

- `python/mlx/spqt.py` — port from
  `extensions/mlx_spqt/mlx_spqt/quantization.py`.

## Step 1 — nanobind bindings in `python/src/ops.cpp`

Add the two bindings right after the `quantized_matmul` binding
ends at line 4351 (before `m.def("quantize", ...)` at line 4352).
Pattern mirrors `quantized_matmul`:

```cpp
  m.def(
      "zigzag_qmv_dense",
      &mx::zigzag_qmv_dense,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zigzag_qmv_dense(x: array, w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        SpQt zigzag-layout dense quantized GEMV.

        Same logical math as :func:`quantized_matmul` but on the SpQt
        zigzag layout. See https://arxiv.org/abs/2511.04477.

        Args:
          x (array): Input activations of shape ``(B, K)``, dtype
            float16.
          w (array): Zigzag-packed quantized weights of shape
            ``(M/group_size, K, group_size*bits/32)``, dtype uint32.
          scales (array): Per-group scales of shape ``(M/group_size,
            K)``, dtype float16.
          biases (array): Per-group biases of shape ``(M/group_size,
            K)``, dtype float16.
          group_size (int, optional): Quantization group size /
            row-band size. Default: ``64``.
          bits (int, optional): Bits per quantized weight. Default:
            ``4``.

        Returns:
          array: Result of ``x @ dequantize(w).T``, shape ``(B, M)``,
          dtype float32.
      )pbdoc");
  m.def(
      "zigzag_qmv_sparse",
      &mx::zigzag_qmv_sparse,
      nb::arg(),
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "num_simdgroups"_a = 2,
      "threadgroups_per_band"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zigzag_qmv_sparse(x: array, sparse_indices: array, w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, num_simdgroups: int = 2, threadgroups_per_band: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        SpQt zigzag-layout sparse quantized GEMV.

        Skips K-positions not present in ``sparse_indices``. Kernel
        runtime scales with ``len(sparse_indices)``, not ``K`` —
        this is the SpQt value proposition realized.

        Args:
          x (array): Input activations of shape ``(B, K)``, dtype
            float16.
          sparse_indices (array): Selected K-positions in count-prefix
            format ``[n, idx_0, ..., idx_{n-1}]``, dtype int32.
          w (array): Zigzag-packed quantized weights, dtype uint32.
          scales (array): Per-group scales, dtype float16.
          biases (array): Per-group biases, dtype float16.
          group_size (int, optional): Default: ``64``.
          bits (int, optional): Default: ``4``.
          num_simdgroups (int, optional): Threadgroup geometry —
            simdgroups per threadgroup. Default: ``2``.
          threadgroups_per_band (int, optional): Threadgroup
            geometry — threadgroups per row-band. Default: ``4``.

        Returns:
          array: ``x[:, idx] @ dequantize(w)[:, idx].T``, shape
          ``(B, M)``, dtype float32.
      )pbdoc");
```

Notes vs `quantized_matmul`:
- **No defaults of `nb::none()` for `scales` / `biases`** —
  quantized_matmul makes biases optional because it supports modes
  where biases aren't used (like mxfp4). Our op always needs biases
  (affine zigzag).
- **`group_size`/`bits` default to concrete `64` / `4`**, not
  `nb::none()`. The MVP only supports those values; failing fast
  on unsupported values is cleaner than silent fallback.
- **`nb::arg()` count for sparse**: 3 positional (x, sparse_indices,
  w) before keyword-style args.

## Step 2 — Create `python/mlx/spqt.py`

Direct port from `extensions/mlx_spqt/mlx_spqt/quantization.py`.
Just change the module docstring and drop the `mlx_spqt` references.

```python
# Copyright © 2026 Apple Inc.

"""SpQt zigzag-layout quantization helpers.

Provides ``quantize_zigzag`` / ``dequantize_zigzag`` — fp-space
weight rearrangements on top of ``mx.quantize`` / ``mx.dequantize``.
The kernel ops ``mx.zigzag_qmv_dense`` and ``mx.zigzag_qmv_sparse``
operate on weights produced by ``quantize_zigzag``.

See https://arxiv.org/abs/2511.04477 for the SpQt algorithm.
"""

import mlx.core as mx


def quantize_zigzag(w, group_size=64, bits=4):
    """Affine-zigzag quantization. Mirrors ``mx.quantize(w, mode='affine')`` style.

    Rearranges ``w`` in fp space so that ``group_size`` output rows × 1 input
    column become contiguous along the last axis, then applies standard
    affine quantization. The result is bit-equivalent to what would be
    produced by a future ``mx.quantize(w, group_size, bits, mode='affine_zigzag')``.

    Args:
        w (mx.array): shape ``(M, K)``. ``M`` must be divisible by
            ``group_size``.
        group_size (int): Quantization group size = row-band size
            (locked equal in zigzag). Default: ``64``.
        bits (int): Bit-width. MVP supports ``4`` only.

    Returns:
        Tuple of three arrays:
            ``w_q``: ``(M/group_size, K, group_size * bits / 32)`` uint32
            ``scales``: ``(M/group_size, K)`` ``w.dtype``
            ``biases``: ``(M/group_size, K)`` ``w.dtype``
    """
    M, K = w.shape
    assert M % group_size == 0, (
        f"M={M} must be divisible by group_size={group_size}"
    )

    # fp-space rearrangement: (M, K) → (M/group_size, K, group_size)
    w_rearr = w.reshape(M // group_size, group_size, K).transpose(0, 2, 1)

    # standard mx.quantize along last axis with size group_size
    w_q, scales, biases = mx.quantize(w_rearr, group_size, bits=bits, mode="affine")
    return w_q, scales.squeeze(-1), biases.squeeze(-1)


def dequantize_zigzag(w_q, scales, biases, group_size=64, bits=4):
    """Inverse of :func:`quantize_zigzag`.

    Returns:
        mx.array: shape ``(M, K)`` where ``M = w_q.shape[0] * group_size``,
        ``K = w_q.shape[1]``.
    """
    # re-add the trailing-1 group dim that mx.dequantize expects
    scales_3d = scales[..., None]
    biases_3d = biases[..., None]

    # standard dequantize returns (M/group_size, K, group_size)
    w_rearr = mx.dequantize(
        w_q, scales_3d, biases_3d, group_size=group_size, bits=bits, mode="affine"
    )

    # inverse rearrangement: (M/group_size, K, group_size) → (M, K)
    n_bands, K, _ = w_rearr.shape
    return w_rearr.transpose(0, 2, 1).reshape(n_bands * group_size, K)
```

## Step 3 — Build and verify

```bash
pip install -e .
```

Quick smoke test from Python:

```python
import mlx.core as mx
from mlx.spqt import quantize_zigzag, dequantize_zigzag

# Round-trip first
w = mx.random.normal(shape=(64, 256)).astype(mx.float16)
w_q, sc, bs = quantize_zigzag(w)
w_dq = dequantize_zigzag(w_q, sc, bs)
print("quantize/dequantize round-trip:", (w - w_dq).abs().max().item())
# Expect ~0.05 or so (4-bit quantization noise)

# Then dense kernel
x = mx.random.normal(shape=(1, 256)).astype(mx.float16)
y = mx.zigzag_qmv_dense(x, w_q, sc, bs)
y_ref = x.astype(mx.float32) @ w_dq.astype(mx.float32).T
print("dense kernel error:", (y - y_ref).abs().max().item())
# Expect < 1e-3

# Then sparse kernel
import numpy as np
idx_np = np.sort(np.random.choice(256, 128, replace=False)).astype(np.int32)
idx_np = np.insert(idx_np, 0, 128)  # count-prefix
idx = mx.array(idx_np)
y_s = mx.zigzag_qmv_sparse(x, idx, w_q, sc, bs)
y_s_ref = x.astype(mx.float32)[:, idx[1:]] @ w_dq.astype(mx.float32)[:, idx[1:]].T
print("sparse kernel error:", (y_s - y_s_ref).abs().max().item())
# Expect < 1e-3
```

If all three errors are within expected ranges, I3 is done and the
ops are fully functional from Python.

## Pitfalls

1. **Argument order in `nb::sig` must match the C++ signature
   order exactly**. Specifically for sparse: positional order is
   `(x, sparse_indices, w, /, scales, biases, ...)`. The `/`
   separates positional-only from keyword-or-positional. Get this
   wrong and Python will raise `TypeError` at call time.

2. **`mx.zigzag_qmv_*` not `mlx.core.zigzag_qmv_*`**: `mx` is
   `mlx.core`, so `m.def("zigzag_qmv_dense", ...)` in
   `python/src/ops.cpp` exposes it at `mlx.core.zigzag_qmv_dense`,
   which is what users import as `mx.zigzag_qmv_dense`.

3. **`python/mlx/spqt.py` works as a submodule without
   `__init__.py`**: the parent `python/mlx/` is a namespace package,
   and adding a single `.py` file creates a submodule reachable
   via `import mlx.spqt`. No need to register anything.

4. **No `.pyi` stub for `spqt.py`**: not needed — the module is
   pure Python, type-checkers will read the source directly.
   (The C++ ops will get stubs automatically via the build's
   stub generator if `MLX_BUILD_PYTHON_STUBS=ON`.)
