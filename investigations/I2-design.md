# I2 — Add C++ op functions in mlx/ops.{cpp,h}

Goal: `mx::zigzag_qmv_dense(...)` and `mx::zigzag_qmv_sparse(...)`
are callable from C++. They wrap the Primitive construction and
output-array building.

After I2, the ops are reachable from C++ but not yet from Python.
Halfway-still-presentable: yes — demonstrates "I integrated this
into MLX core's public C++ API."

## Files to modify

Two files:

- `mlx/ops.h` — declarations near `quantized_matmul` (line 1473).
- `mlx/ops.cpp` — implementations near `quantized_matmul` impl
  (line 4477).

## Step 1 — Declarations in `mlx/ops.h`

Add after the `quantized_matmul` declaration ending at line 1482
(before the `quantize` declaration at line 1485). No change to
existing decls.

```cpp
/** SpQt zigzag-layout dense quantized GEMV. See SpQt paper:
 *  https://arxiv.org/abs/2511.04477 */
MLX_API array zigzag_qmv_dense(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

/** SpQt zigzag-layout sparse quantized GEMV. `sparse_indices` has
 *  format `[n, idx_0, ..., idx_{n-1}]` (count prefixed). Kernel
 *  time scales with `n`, not K. */
MLX_API array zigzag_qmv_sparse(
    const array& x,
    const array& sparse_indices,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size = 64,
    int bits = 4,
    int num_simdgroups = 2,
    int threadgroups_per_band = 4,
    StreamOrDevice s = {});
```

Two notes vs the extension:
- Defaults `(num_simdgroups=2, threadgroups_per_band=4)` for
  sparse — these are the values from the M2 tuning sweep, not
  the extension's old `(4, 4)`.
- Param names mirror `quantized_matmul`: `w` not `w_zz`,
  consistent with how `mx.quantized_matmul(x, w, scales, ...)`
  reads.

## Step 2 — Implementations in `mlx/ops.cpp`

Add after the `quantized_matmul` impl ending at line 4527 (before
the `validate_qqmm_inputs` definition at line 4529).

```cpp
namespace {

void validate_zigzag_inputs(
    const char* op_name,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size,
    int bits) {
  // dtype checks
  if (x.dtype() != float16) {
    std::ostringstream msg;
    msg << "[" << op_name << "] x must be float16, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[" << op_name << "] w must be uint32 (packed), got "
        << w.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scales.dtype() != float16 || biases.dtype() != float16) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] scales and biases must be float16.";
    throw std::invalid_argument(msg.str());
  }
  // bit-width / group-size: only group_size=64, bits=4 in MVP
  if (group_size != 64 || bits != 4) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] only group_size=64 and bits=4 are supported in MVP.";
    throw std::invalid_argument(msg.str());
  }
  // shapes
  // w: (M/gs, K, gs*bits/32) uint32 — first axis is row-bands
  // scales/biases: (M/gs, K) — same n_bands as w
  if (w.ndim() != 3 || scales.ndim() != 2 || biases.ndim() != 2) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] expected w 3D, scales/biases 2D; got "
        << "w.ndim()=" << w.ndim()
        << " scales.ndim()=" << scales.ndim()
        << " biases.ndim()=" << biases.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.shape(0) != scales.shape(0) ||
      w.shape(0) != biases.shape(0)) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] inconsistent n_bands across w/scales/biases.";
    throw std::invalid_argument(msg.str());
  }
  if (x.shape(-1) != w.shape(1)) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] x.shape(-1)=" << x.shape(-1)
        << " must match w.shape(1)=" << w.shape(1) << " (K).";
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace

array zigzag_qmv_dense(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size /* = 64 */,
    int bits /* = 4 */,
    StreamOrDevice s /* = {} */) {
  validate_zigzag_inputs(
      "zigzag_qmv_dense", x, w, scales, biases, group_size, bits);

  const int n_bands = scales.shape(0);
  const int M = n_bands * group_size;
  auto out_shape = x.shape();
  out_shape.back() = M;

  // Inputs ordered to match Primitive's eval_gpu expectations:
  //   inputs[0] = w_zz, [1] = x, [2] = scales, [3] = biases
  return array(
      std::move(out_shape),
      float32,
      std::make_shared<ZigzagQmvDense>(to_stream(s), group_size, bits),
      {w, x, scales, biases});
}

array zigzag_qmv_sparse(
    const array& x,
    const array& sparse_indices,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size /* = 64 */,
    int bits /* = 4 */,
    int num_simdgroups /* = 2 */,
    int threadgroups_per_band /* = 4 */,
    StreamOrDevice s /* = {} */) {
  validate_zigzag_inputs(
      "zigzag_qmv_sparse", x, w, scales, biases, group_size, bits);
  if (sparse_indices.dtype() != int32) {
    std::ostringstream msg;
    msg << "[zigzag_qmv_sparse] sparse_indices must be int32, got "
        << sparse_indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  const int n_bands = scales.shape(0);
  const int M = n_bands * group_size;
  auto out_shape = x.shape();
  out_shape.back() = M;

  // inputs[0]=w_zz, [1]=x, [2]=scales, [3]=biases, [4]=sparse_indices
  return array(
      std::move(out_shape),
      float32,
      std::make_shared<ZigzagQmvSparse>(
          to_stream(s),
          group_size,
          bits,
          num_simdgroups,
          threadgroups_per_band),
      {w, x, scales, biases, sparse_indices});
}
```

## Step 3 — Build and verify

```bash
pip install -e .
```

Expected: `mlx_core` builds. Both `zigzag_qmv_*` symbols are now
public C++ API. There's still no Python binding (that's I3), so
testing from Python isn't possible yet.

If you want a quick C++-side smoke check, you could write a tiny
test executable, but it's probably faster to just move to I3 and
test from Python.

## Pitfalls

1. **Input ordering must match `eval_gpu`'s `inputs[N]` indexing**.
   In the Primitive's `eval_gpu`:
   - Dense: `inputs[0]=w_zz, [1]=x, [2]=scales, [3]=biases`
   - Sparse: same plus `[4]=sparse_indices`

   The op function builds the input vector in the same order. If
   you swap `x` and `w` in the brace-list, runtime will crash with
   confusing memory errors.

2. **`to_stream(s)`**: this is the helper that converts
   `StreamOrDevice` to `Stream`. Used by every op in `ops.cpp` —
   no need to re-implement.

3. **`out_shape = x.shape()` then mutate `back()`**: mirrors how
   `quantized_matmul` does it (`ops.cpp:4519-4520`). This handles
   the batch dimension automatically (shape becomes `(B, M)`).

4. **Don't add `mode` / `transpose` parameters**: `quantized_matmul`
   has them because it supports multiple modes and transpose=False.
   We only have one layout (zigzag) and the kernel always assumes
   the transpose-equivalent semantic. Keep our signature minimal.
