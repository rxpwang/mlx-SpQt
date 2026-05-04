# I0 — MLX core integration: structural map

Goal: identify the exact file-level changes needed to move
`zigzag_qmv_dense` / `zigzag_qmv_sparse` from the standalone extension
under `extensions/mlx_spqt/` into MLX core, so they're reachable as
`mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse`.

Scope: **Metal-only standalone integration.** No CPU fallback, no
CUDA, no mode-dispatch. Mirrors the `quantized_matmul` layout because
`quantized_matmul` is the closest existing op (same primitive shape:
takes packed weights + scales + biases, dispatches a Metal kernel).

## The five layers of an MLX op

`quantized_matmul` exists at five layers; we need to add a parallel
entry for `zigzag_qmv_dense` and `zigzag_qmv_sparse` at each:

| Layer | What | quantized_matmul location |
|---|---|---|
| 1. **Op function decl** | C++ public API | `mlx/ops.h:1473` |
| 2. **Op function impl** | shape/dtype validation, build Primitive | `mlx/ops.cpp:4477` |
| 3. **Primitive class** | UnaryPrimitive subclass | `mlx/primitives.h:1615` |
| 4. **Primitive impls** | `is_equivalent`, `output_shapes`, vmap/vjp/jvp | `mlx/primitives.cpp:3548` |
| 5. **Backend eval_gpu** | Metal kernel dispatch | `mlx/backend/metal/quantized.cpp:1387` |
| 6. **Metal kernels** | MSL templates + instantiations | `mlx/backend/metal/kernels/quantized.{h,metal}` |
| 7. **Python binding** | nanobind export | `python/src/ops.cpp:4312` |

## Target layout for our zigzag ops

### New files

| Path | What |
|---|---|
| `mlx/backend/metal/kernels/quantized_zigzag.h` | MSL kernel templates (port of `extensions/mlx_spqt/zigzag_qmv_*/zigzag_qmv_*.metal`'s template body). |
| `mlx/backend/metal/kernels/quantized_zigzag.metal` | `host_name` instantiation list (port of the `instantiate_kernel` block in the extension's `.metal` files). |
| `mlx/backend/metal/quantized_zigzag.cpp` | `ZigzagQmvDense::eval_gpu` and `ZigzagQmvSparse::eval_gpu` (port of `extensions/mlx_spqt/zigzag_qmv_*/zigzag_qmv_*.cpp`'s `eval_gpu` body). |
| `python/mlx/spqt.py` | `quantize_zigzag`, `dequantize_zigzag` (Python helpers), plus optional re-exports of `mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse`. Direct port of `extensions/mlx_spqt/mlx_spqt/quantization.py`. |

### Existing files to modify

| Path | What to add |
|---|---|
| `mlx/ops.h` | Two new declarations near `quantized_matmul` decl at `:1473`. |
| `mlx/ops.cpp` | Two new op fn impls near `quantized_matmul` at `:4477`. Validation, `make_shared<ZigzagQmvDense>`, return `array(...)`. |
| `mlx/primitives.h` | Two new Primitive classes near `QuantizedMatmul` at `:1615`. State: `(group_size, bits)` for dense; same plus index-array passing for sparse. |
| `mlx/primitives.cpp` | `is_equivalent`, `output_shapes` for both. Skip vmap/vjp/jvp (use `DEFINE_VMAP()` / `DEFINE_GRADS()` macros that throw NYI). |
| `mlx/backend/metal/CMakeLists.txt` | After `:79` (`make_jit_source(quantized ...)`): add `make_jit_source(quantized_zigzag kernels/quantized_utils.h)`. After `:126` (`${CMAKE_CURRENT_SOURCE_DIR}/quantized.cpp`): add `${CMAKE_CURRENT_SOURCE_DIR}/quantized_zigzag.cpp`. |
| `mlx/backend/metal/kernels/CMakeLists.txt` | After `:138` (`build_kernel(quantized ...)`): add `build_kernel(quantized_zigzag quantized_zigzag.h ...)`. |
| `python/src/ops.cpp` | Two new `m.def(...)` entries near `quantized_matmul` at `:4312`. |
| `python/tests/test_quantized.py` (or new `test_quantized_zigzag.py`) | Pytest cases mirroring `extensions/mlx_spqt/test_zigzag_qmv_*.py`. |

### Out of scope (stay as-is)

- **`quantize_zigzag` / `dequantize_zigzag`**: pure Python wrappers
  (reshape + `mx.quantize`). They're 50 lines, don't dispatch a
  custom kernel, and would be more friction than value in C++.
  **Move them into MLX's Python package as `python/mlx/spqt.py`**
  so `from mlx.spqt import quantize_zigzag, dequantize_zigzag`
  works. This mirrors how `mlx.utils`, `mlx.nn`, `mlx.optimizers`
  live as submodules alongside the compiled `mlx.core`. The C++
  kernel ops stay at `mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse`
  in `mlx.core` — same layer as `mx.quantized_matmul` — and are
  *not* re-exported from `mlx.spqt`.
- **CPU and CUDA backends**: explicit non-goal. `eval_cpu` is not
  implemented in MLX-SpQt either; `DEFINE_NO_CPU_PRIMITIVE` or
  `throw std::runtime_error("NYI")` from `eval_cpu` is fine for the
  Primitive.

## Detailed per-phase plan

### I1 — Move kernel + Primitive files (2-3 hr, halfway-stop OK)

Concrete checklist:

1. **Create `mlx/backend/metal/kernels/quantized_zigzag.h`**.
   - Copy template body from `extensions/mlx_spqt/zigzag_qmv_dense/zigzag_qmv_dense.metal`'s template section. Same for sparse.
   - Replace any `mlx_spqt::` names; use the same naming convention as `quantized.h` (e.g., `affine_zigzag_qmv_dense<T, group_size, bits, ...>`).
   - Include `quantized_utils.h` for shared helpers (qdot, etc.).
2. **Create `mlx/backend/metal/kernels/quantized_zigzag.metal`**.
   - Copy `host_name` instantiation list from the extension's metal files. Same instantiations: 11 sparse combos covering (NSG, TG) ∈ {(2,2), (2,4), (4,1), (4,2), (4,4), (4,8), (8,2), (1,4), (2,1), (2,8), (8,1)}; dense the same default it currently uses.
3. **Create `mlx/backend/metal/quantized_zigzag.cpp`**.
   - Port `eval_gpu` bodies from `extensions/mlx_spqt/zigzag_qmv_*/zigzag_qmv_*.cpp`.
   - Use MLX core's helpers: `metal::device(s.device)`, `ensure_row_contiguous_matrix`, `get_kernel`, etc. Mirror what `mlx/backend/metal/quantized.cpp:1387` (`QuantizedMatmul::eval_gpu`) does.
4. **Add ZigzagQmvDense / ZigzagQmvSparse Primitive classes** in `mlx/primitives.h` (near `:1615`). Skeleton:
   ```cpp
   class ZigzagQmvDense : public UnaryPrimitive {
    public:
     explicit ZigzagQmvDense(Stream stream, int group_size, int bits)
         : UnaryPrimitive(stream), group_size_(group_size), bits_(bits) {}
     void eval_cpu(const std::vector<array>&, array&) override {
       throw std::runtime_error("[ZigzagQmvDense] CPU NYI.");
     }
     void eval_gpu(const std::vector<array>&, array&) override;
     DEFINE_VMAP(); DEFINE_GRADS(); DEFINE_NAME(ZigzagQmvDense)
     bool is_equivalent(const Primitive& other) const override;
     std::vector<Shape> output_shapes(const std::vector<array>&) override;
    private: int group_size_; int bits_;
   };
   ```
   ZigzagQmvSparse is the same plus the idx is passed as input array (not state).
5. **`is_equivalent` / `output_shapes`** in `mlx/primitives.cpp` (near `:3548`).
6. **Wire CMake**: add `make_jit_source` line, add source line, add `build_kernel` line. See locations in the table above.
7. **Build**: `pip install -e .` from repo root. Verify `mlx_core` builds and links. **At this stop point, the kernel code lives in core, builds, but is not yet callable from Python.** Halfway-presentable: yes — demonstrates "I know where this code goes upstream."

### I2 — Add C++ op functions (1-2 hr)

1. **Declarations in `mlx/ops.h`** (after `:1483`):
   ```cpp
   MLX_API array zigzag_qmv_dense(
       array x, array w, array scales, array biases,
       int group_size = 64, int bits = 4,
       StreamOrDevice s = {});
   MLX_API array zigzag_qmv_sparse(
       array x, array idx, array w, array scales, array biases,
       int group_size = 64, int bits = 4,
       StreamOrDevice s = {});
   ```
2. **Impls in `mlx/ops.cpp`** (after `:4527`).
   - Validate dtypes (fp16/fp32 for x/scales/biases, uint32 for w, int32 for idx).
   - Compute output shape: `(B, M)` where M is `w.shape[0] * group_size`.
   - Build the Primitive with `make_shared<ZigzagQmvDense>(...)`.
   - Return `array(out_shape, out_dtype, primitive, inputs)`.

   Reference: `quantized_matmul` impl at `mlx/ops.cpp:4477`.
3. **Build**: `pip install -e .` again. The op is now callable from C++ but not yet from Python.

### I3 — Python bindings + Python helpers (1.5-2.5 hr)

1. **C++ ops nanobind bindings**. Add to `python/src/ops.cpp`
   (near `:4312`, the `quantized_matmul` binding):
   ```cpp
   m.def("zigzag_qmv_dense", &mx::zigzag_qmv_dense,
         "x"_a, "w"_a, "scales"_a, "biases"_a,
         nb::kw_only(), "group_size"_a = 64, "bits"_a = 4,
         "stream"_a = nb::none(),
         nb::sig("def zigzag_qmv_dense(x: array, w: array, /, scales: array, biases: array, *, group_size: int = 64, bits: int = 4, stream: Union[None, Stream, Device] = None) -> array"),
         "Sparse-Promoted Quantization (zigzag) dense GEMV — see SpQt paper.");
   ```
   Same shape for sparse, with `idx` as positional arg.
2. **Python helpers**. Create `python/mlx/spqt.py` by porting
   `extensions/mlx_spqt/mlx_spqt/quantization.py` verbatim
   (50 lines: `quantize_zigzag` + `dequantize_zigzag`). No
   re-exports of the C++ ops — those stay at `mx.zigzag_qmv_*` so
   the API has one canonical path per op.
3. **Build**: `pip install -e .` from repo root. Verify both
   `mx.zigzag_qmv_dense` (C++ binding) and `from mlx.spqt import
   quantize_zigzag` (Python module) work.

### I4 — Move tests (1-2 hr; can run in parallel with I5)

Add `python/tests/test_quantized_zigzag.py`:
- Port `extensions/mlx_spqt/test_zigzag_qmv_dense.py` and
  `test_zigzag_qmv_sparse.py` to `mlx_tests.MLXTestCase` style
  (mirror `python/tests/test_quantized.py:11` `test_quantize_dequantize`).
- Use `mx.zigzag_qmv_dense` / `mx.zigzag_qmv_sparse` instead of
  `mlx_spqt.*`.
- Same shape coverage (7 shapes × 2 densities for sparse).

### I5 — Smoke-validate headline numbers (30 min)

- Run the demo (or a stripped version) against the integrated
  `mx.*` API. Confirm correctness passes and ~1.8× speedup at LLM-FFN
  shapes still holds.
- If numbers are off by >5%, suspect a host-side regression
  (kernel pipeline cache, etc.) — bisect against the extension
  baseline.

## Key risks

1. **JIT vs nojit code path**. MLX has both
   `mlx/backend/metal/jit_kernels.cpp` (JIT) and `nojit_kernels.cpp`
   paths. The extension bypasses JIT and registers a static library.
   Core uses JIT. Action: verify `make_jit_source(quantized_zigzag …)`
   in I1 step 6, and check whether the kernel needs entries in
   `jit_kernels.cpp` too (likely yes for the templated instantiations).
2. **`get_quantized_kernel` lookup pattern**. Core's
   `quantized.cpp:20` (`get_quantized_kernel_wrapped`) uses a different
   kernel-registration model than the extension's
   `mx::concatenate`-based name-construction. Need to mirror it or
   route around it. Action: probably easiest to add a parallel
   `get_quantized_zigzag_kernel` helper rather than overload the
   existing one.
3. **Naming collision with the extension**. The standalone extension
   under `extensions/mlx_spqt/` will continue to coexist (we don't
   tear it down). Both define a `ZigzagQmvDense` class. They live in
   separate translation units / namespaces, so this isn't a runtime
   collision, but it's worth being aware that the integration is
   *parallel*, not a *replacement*.

## Files to read for I1 design

Before starting I1, read these in order:

1. `mlx/backend/metal/quantized.cpp:1387-1500` —
   `QuantizedMatmul::eval_gpu` body. This is the template I1 step 3
   should follow.
2. `mlx/backend/metal/kernels/quantized.h` (pick any one
   `affine_qmv_*` kernel) — kernel template body shape.
3. `mlx/backend/metal/kernels/quantized.metal` — instantiation list
   format.
4. `extensions/mlx_spqt/zigzag_qmv_dense/zigzag_qmv_dense.cpp` and
   `.metal` — what we're porting from.
