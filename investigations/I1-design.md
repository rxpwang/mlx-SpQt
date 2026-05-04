# I1 — Move kernel + Primitive into mlx/backend/metal/

Goal: kernel files live in MLX core, the project builds, the
zigzag ops are reachable from C++ even if not yet from Python.

Two key design decisions resolved up front (different from the
extension's choices):

1. **`UnaryPrimitive` instead of `Primitive`** — both ops have a
   single output. `UnaryPrimitive` (mlx/primitives.h:127) is a thin
   wrapper that adapts the single-output signature to the multi-
   output Primitive interface. Cleaner `eval_gpu(inputs, out&)`
   signature; matches every other quantized op in core.
2. **NoJIT path only** — MLX has both a JIT path (where templates
   live in `kernels/*.h` and get preprocessed at runtime) and a
   noJIT path (where `kernels/*.metal` is precompiled into a
   `.metallib` at build time). NoJIT is the default. We only wire
   noJIT for I1; JIT support is deferred to a polish step.

## File summary

### Create

| Path | Source | What |
|---|---|---|
| `mlx/backend/metal/kernels/quantized_zigzag.h` | from `extensions/mlx_spqt/zigzag_qmv_dense/zigzag_qmv_dense.metal` (template body L23-end) and `zigzag_qmv_sparse.metal` (template body L23-end) | Template impls + zero-fill kernel. |
| `mlx/backend/metal/kernels/quantized_zigzag.metal` | from extension's `.metal` files (the `[[host_name(...)]]` instantiation blocks) | Instantiation list. Includes `quantized_zigzag.h`. |
| `mlx/backend/metal/quantized_zigzag.cpp` | from `extensions/mlx_spqt/zigzag_qmv_dense/zigzag_qmv_dense.cpp` and `zigzag_qmv_sparse.cpp` (eval_gpu bodies) | `eval_gpu` for both Primitives. |

### Modify

| Path | What |
|---|---|
| `mlx/primitives.h` | Add `ZigzagQmvDense`, `ZigzagQmvSparse` classes (skeleton below). |
| `mlx/primitives.cpp` | Add `is_equivalent` and `output_shapes` (skeleton below). |
| `mlx/backend/metal/CMakeLists.txt` | One line: source list. |
| `mlx/backend/metal/kernels/CMakeLists.txt` | One line: `build_kernel`. |

---

## Step-by-step

### Step 1 — Create `kernels/quantized_zigzag.h`

This is the templated kernel implementation. Port from the
extension's two `.metal` files:

```cpp
// Copyright © 2026 Apple Inc.

#pragma once

// (Note: when this file is consumed by quantized_zigzag.metal, all
//  these includes are already present via that .metal file. We
//  still include them here so the .h is self-contained for IDE/JIT.)

template <typename T, int group_size, int bits, int num_simdgroups, int threadgroups_per_band>
[[kernel]] void zigzag_qmv_dense_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {
  // Body: copy verbatim from
  //   extensions/mlx_spqt/zigzag_qmv_dense/zigzag_qmv_dense.metal
  // (the function body, not the [[host_name]] instantiations).
}

template <typename T, int group_size, int bits, int num_simdgroups, int threadgroups_per_band>
[[kernel]] void zigzag_qmv_sparse_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {
  // Body: copy verbatim from
  //   extensions/mlx_spqt/zigzag_qmv_sparse/zigzag_qmv_sparse.metal
}

[[kernel]] void zigzag_zero_fill_float(
    device float* out [[buffer(0)]],
    uint id [[thread_position_in_grid]]) {
  out[id] = 0.0f;
}
```

Note: `zigzag_zero_fill_float` is the un-templated zero-fill helper
already present in `extensions/.../zigzag_qmv_dense.metal:14-19`
(in the sparse file it's commented out — only one definition needed
since it's not templated).

### Step 2 — Create `kernels/quantized_zigzag.metal`

This is the instantiation list. Mirrors `kernels/quantized.metal`'s
structure:

```cpp
// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"  // qdot, get_pack_factor, get_bytes_per_pack
#include "mlx/backend/metal/kernels/quantized_zigzag.h"

// === Dense instantiations ===
template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 4, 4>(
    device const uint32_t*  w_zz      [[buffer(0)]],
    device const half*      x         [[buffer(1)]],
    device const half*      scales_zz [[buffer(2)]],
    device const half*      biases_zz [[buffer(3)]],
    device atomic<float>*   out       [[buffer(4)]],
    constant const int&     K         [[buffer(5)]],
    uint3  tid       [[threadgroup_position_in_grid]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]]);

// === Sparse instantiations (11 NSG×TG combinations) ===
// (2,2), (2,4), (4,1), (4,2), (4,4), (4,8), (8,2), (1,4), (2,1), (2,8), (8,1)
// Copy these verbatim from
//   extensions/mlx_spqt/zigzag_qmv_sparse/zigzag_qmv_sparse.metal
// adjusting the kernel name / template params only.
```

Note: only one dense instantiation (NSG=4, TG=4) is needed since
the dense kernel's geometry is hard-coded in `eval_gpu` (the
extension uses `constexpr int num_simdgroups = 4; threadgroups_per_band = 4;`).

### Step 3 — Create `mlx/backend/metal/quantized_zigzag.cpp`

Port `eval_gpu` for both Primitives. Combine into one file (mirrors
how `quantized.cpp` holds both `QuantizedMatmul::eval_gpu` and
`GatherQMM::eval_gpu`):

```cpp
// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

inline std::string zigzag_kernel_name(
    const std::string& base,
    const std::string& type,
    int group_size,
    int bits,
    int num_simdgroups,
    int threadgroups_per_band) {
  std::string name;
  name.reserve(64);
  concatenate(
      name, base, "_", type,
      "_gs_", group_size,
      "_b_", bits,
      "_nsg_", num_simdgroups,
      "_tg_", threadgroups_per_band);
  return name;
}

void zero_fill_float_buffer(
    const Stream& s, metal::Device& d, array& out) {
  auto zero_kernel = d.get_kernel("zigzag_zero_fill_float");
  auto& zfe = metal::get_command_encoder(s);
  zfe.set_compute_pipeline_state(zero_kernel);
  zfe.set_output_array(out, 0);
  const size_t n = out.size();
  const size_t tgp = std::min(n, zero_kernel->maxTotalThreadsPerThreadgroup());
  zfe.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(tgp, 1, 1));
}

}  // namespace

void ZigzagQmvDense::eval_gpu(
    const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& w_zz   = inputs[0];
  auto& x      = inputs[1];
  auto& scales = inputs[2];
  auto& biases = inputs[3];

  constexpr int num_simdgroups = 4;
  constexpr int threadgroups_per_band = 4;
  constexpr int simdgroup_size = 32;

  zero_fill_float_buffer(s, d, out);

  auto kname = zigzag_kernel_name(
      "zigzag_qmv_dense", "half",
      group_size_, bits_, num_simdgroups, threadgroups_per_band);
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(w_zz, 0);
  enc.set_input_array(x, 1);
  enc.set_input_array(scales, 2);
  enc.set_input_array(biases, 3);
  enc.set_output_array(out, 4);
  const int K = x.shape(-1);
  enc.set_bytes(K, 5);

  const int M = out.shape(-1);
  MTL::Size group_dims(num_simdgroups * simdgroup_size, 1, 1);
  MTL::Size grid_dims(M / group_size_, threadgroups_per_band, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

void ZigzagQmvSparse::eval_gpu(
    const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& w_zz       = inputs[0];
  auto& x          = inputs[1];
  auto& scales     = inputs[2];
  auto& biases     = inputs[3];
  auto& sparse_idx = inputs[4];

  constexpr int simdgroup_size = 32;

  zero_fill_float_buffer(s, d, out);

  auto kname = zigzag_kernel_name(
      "zigzag_qmv_sparse", "half",
      group_size_, bits_, num_simdgroups_, threadgroups_per_band_);
  auto kernel = d.get_kernel(kname);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(w_zz, 0);
  enc.set_input_array(x, 1);
  enc.set_input_array(scales, 2);
  enc.set_input_array(biases, 3);
  enc.set_input_array(sparse_idx, 4);
  enc.set_output_array(out, 5);
  const int K = x.shape(-1);
  enc.set_bytes(K, 6);

  const int M = out.shape(-1);
  MTL::Size group_dims(num_simdgroups_ * simdgroup_size, 1, 1);
  MTL::Size grid_dims(M / group_size_, threadgroups_per_band_, 1);
  enc.dispatch_threadgroups(grid_dims, group_dims);
}

}  // namespace mlx::core
```

Two notes vs the extension version:
- **No `get_library` / `current_binary_dir`** — kernels live in the
  bundled `mlx.metallib` and are fetched by name.
- **`d.get_kernel(name)` not `d.get_kernel(name, lib)`** — same.

### Step 4 — Add Primitive classes in `mlx/primitives.h`

Add right after the `QuantizedMatmul` class (currently ends at
line 1646). Skeleton:

```cpp
class ZigzagQmvDense : public UnaryPrimitive {
 public:
  explicit ZigzagQmvDense(Stream stream, int group_size, int bits)
      : UnaryPrimitive(stream),
        group_size_(group_size),
        bits_(bits) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("[ZigzagQmvDense] CPU NYI.");
  }
  void eval_gpu(const std::vector<array>&, array&) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_NAME(ZigzagQmvDense)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

 private:
  int group_size_;
  int bits_;
};

class ZigzagQmvSparse : public UnaryPrimitive {
 public:
  explicit ZigzagQmvSparse(
      Stream stream, int group_size, int bits,
      int num_simdgroups, int threadgroups_per_band)
      : UnaryPrimitive(stream),
        group_size_(group_size),
        bits_(bits),
        num_simdgroups_(num_simdgroups),
        threadgroups_per_band_(threadgroups_per_band) {}

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("[ZigzagQmvSparse] CPU NYI.");
  }
  void eval_gpu(const std::vector<array>&, array&) override;

  DEFINE_VMAP()
  DEFINE_GRADS()
  DEFINE_NAME(ZigzagQmvSparse)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;

 private:
  int group_size_;
  int bits_;
  int num_simdgroups_;
  int threadgroups_per_band_;
};
```

`DEFINE_VMAP()` and `DEFINE_GRADS()` are the canonical macros for
"this primitive doesn't support vmap/vjp/jvp"; they expand to
methods that throw NYI. Confirmed by `QuantizedMatmul` using them
(primitives.h:1632-1633).

### Step 5 — Add Primitive method bodies in `mlx/primitives.cpp`

Add right after the `QuantizedMatmul::output_shapes` (currently ends
around line 3561):

```cpp
bool ZigzagQmvDense::is_equivalent(const Primitive& other) const {
  const ZigzagQmvDense& o = static_cast<const ZigzagQmvDense&>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_;
}

std::vector<Shape> ZigzagQmvDense::output_shapes(
    const std::vector<array>& inputs) {
  // inputs = {w_zz, x, scales, biases}
  // out shape: (B, n_bands * group_size) where n_bands = scales.shape(0)
  const int n_bands = inputs[2].shape(0);
  const int M = n_bands * group_size_;
  auto out_shape = inputs[1].shape();  // (B, K) → (B, M)
  out_shape.back() = M;
  return {std::move(out_shape)};
}

bool ZigzagQmvSparse::is_equivalent(const Primitive& other) const {
  const ZigzagQmvSparse& o = static_cast<const ZigzagQmvSparse&>(other);
  return group_size_ == o.group_size_ && bits_ == o.bits_
      && num_simdgroups_ == o.num_simdgroups_
      && threadgroups_per_band_ == o.threadgroups_per_band_;
}

std::vector<Shape> ZigzagQmvSparse::output_shapes(
    const std::vector<array>& inputs) {
  // inputs = {w_zz, x, scales, biases, sparse_indices}
  const int n_bands = inputs[2].shape(0);
  const int M = n_bands * group_size_;
  auto out_shape = inputs[1].shape();
  out_shape.back() = M;
  return {std::move(out_shape)};
}
```

### Step 6 — Wire CMake

**`mlx/backend/metal/CMakeLists.txt`**: add one line in the
`target_sources` block (around line 126, after `quantized.cpp`):

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp
${CMAKE_CURRENT_SOURCE_DIR}/quantized_zigzag.cpp   # ← add this
```

**`mlx/backend/metal/kernels/CMakeLists.txt`**: add one line in the
`if(NOT MLX_METAL_JIT)` block (around line 138, after the existing
`build_kernel(quantized ...)`):

```cmake
build_kernel(quantized quantized.h quantized_utils.h ${STEEL_HEADERS})
build_kernel(quantized_zigzag quantized_zigzag.h quantized.h quantized_utils.h ${STEEL_HEADERS})  # ← add this
```

The dependency list (`quantized.h quantized_utils.h ${STEEL_HEADERS}`)
tells CMake to rebuild the kernel if any of those headers changes.
We include `quantized.h` because our `.metal` includes it for `qdot`.

### Step 7 — Build and verify

```bash
pip install -e .
```

Expected outcome:
- `mlx_core` library builds.
- `quantized_zigzag` kernel symbols appear in the bundled metallib.
- The C++ `ZigzagQmvDense` and `ZigzagQmvSparse` classes are
  defined; `is_equivalent` and `output_shapes` link.
- **Not yet callable**: there's no op function (I2) and no Python
  binding (I3). That's expected — this is the "halfway-still-
  presentable" stop point.

Verify the metallib has the new kernel:

```bash
# After build, find the bundled metallib and inspect symbols
find . -name "mlx.metallib" 2>/dev/null
xcrun metal-nm <path-to-mlx.metallib> | grep zigzag_qmv | head -5
```

Expected: 12 symbols (1 dense + 11 sparse) starting with
`zigzag_qmv_*` and the `zigzag_zero_fill_float` symbol.

---

## Common pitfalls to watch for

1. **`#include "mlx/primitives.h"` in `quantized_zigzag.cpp`**:
   needed so `ZigzagQmvDense::eval_gpu` resolves to the class
   declared there. Without it, the linker will complain about the
   class being undefined.

2. **Forgetting `quantized.h` in the `build_kernel` dep list**:
   the `.metal` file uses `qdot` and `get_pack_factor` from
   `quantized.h`; CMake won't rebuild the kernel after edits to
   `quantized.h` unless it's in the dep list.

3. **`DEFINE_VMAP()` and `DEFINE_GRADS()` macros only DECLARE
   methods, they don't define throw-NYI bodies** (verified at
   `primitives.h:13-28`). You must add the bodies in
   `primitives.cpp` — see how `QuantizedMatmul::vmap` /
   `vjp` / `jvp` are defined at `primitives.cpp:3453`/`3459`/`3527`.
   If you skip this, the linker will complain about undefined
   vtable symbols for `vmap`/`vjp`/`jvp`.

4. **`namespace mlx::core` vs `namespace spqt_ext`**: change all
   namespace references when porting from the extension.

5. **`mx::array` → `array`**: inside `namespace mlx::core`, drop the
   `mx::` prefix.

6. **Zero-fill kernel collision**: the extension defines
   `zigzag_zero_fill_float` in both dense and sparse `.metal` files
   (commented out in sparse). In core, define it exactly once — in
   `quantized_zigzag.h`. If the symbol appears twice the metallib
   build will fail with a duplicate-symbol error.
