# MLX extension internals: the axpby walkthrough

A code-anchored study of `examples/extensions/axpby/` — MLX's canonical extension
example — followed in execution order. Used as the M0b reference for understanding
the extension machinery before we clone the template.

The reading order parallels what happens when a user writes `axpby(x, y, 2., 3.)`:
the call enters Python, crosses into C++, builds a Primitive, lazy-eval'd into a
GPU dispatch, and the kernel runs.

| Order | File / range | Topic | Status |
|---|---|---|---|
| 1 | `bindings.cpp` (all) | Python ↔ C++ binding via nanobind; `__init__.py` re-export | ✅ |
| 2 | `axpby/axpby.cpp:44-75` | The op function — Primitive node construction, returns `mx::array` | ✅ |
| 3 | `axpby/axpby.h` (all) | Primitive class declaration | ✅ |
| 4 | `axpby/axpby.cpp:153-235` | `eval_gpu` — encoder, kernel lookup, buffer binding, dispatch | ✅ |
| 5 | `axpby/axpby.metal` (all) | Kernel templates + `instantiate_kernel` macros | ✅ |
| 6 | `axpby/axpby.cpp:20-31` | `current_binary_dir()` — runtime metallib path resolution | ✅ |
| 7 | `axpby/axpby.cpp:80-144` + `:249-298` | CPU impl + vjp/jvp/vmap | ✅ |

---

## Order 1 — `bindings.cpp` and `__init__.py`

Two short files (39 + 5 lines) that establish the boundary between the C++
implementation and the Python user-facing API.

### Big picture

`bindings.cpp` is **the boundary translator** between C++ and Python. It does one
job: declare which C++ functions are visible from Python, and how arguments cross
the boundary. The full chain when a user calls `axpby(x, y, 2., 3.)`:

```
Python:                              axpby(x, y, 2.0, 3.0)
                                     │
mlx_sample_extensions/__init__.py:5  │   from ._ext import axpby
                                     ▼
mlx_sample_extensions/_ext.so        ._ext.axpby                       ← Python-visible
   (compiled from bindings.cpp)      │
                                     │   nanobind-generated trampoline
                                     │   - Python args → C++ types
                                     │   - mx.core.array → mx::array
                                     ▼
C++ side, axpby/axpby.cpp:44         my_ext::axpby(x, y, alpha, beta, s)
                                     │   constructs Primitive, returns mx::array
                                     ▼
                                     mx::array → nanobind → mlx.core.array
```

### The binding declaration

```cpp
// bindings.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "axpby/axpby.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Sample extension for MLX";

  m.def(
      "axpby",                        // Python-visible name
      &my_ext::axpby,                 // C++ function pointer
      "x"_a, "y"_a, "alpha"_a, "beta"_a,    // positional arg names
      nb::kw_only(),                  // separator
      "stream"_a = nb::none(),        // kwarg with default None
      R"(... docstring ...)");
}
```

| Construct | What it does |
|---|---|
| `NB_MODULE(_ext, m)` | nanobind macro that expands to: declare a Python module named `_ext`, hand `m` to register things on it. Generates the `PyInit__ext` C entry-point Python's import system calls. |
| `m.doc()` | Sets `_ext.__doc__`. |
| `m.def("axpby", &my_ext::axpby, ...)` | Registers the C++ function as `_ext.axpby`. nanobind reads the C++ signature at compile time and synthesizes a trampoline — a small generated function that takes Python objects, unpacks each into the corresponding C++ type, calls the function, and converts the return back. |
| `"x"_a` | The `_a` literal (from `nb::literals`) creates an argument descriptor with the given name. Without arg descriptors, only positional calls work; with them, kwarg calls work too. |
| `nb::kw_only()` | Separator. Anything after it is keyword-only (cannot be passed positionally). Matches MLX's convention of making `stream` keyword-only. |
| `"stream"_a = nb::none()` | Keyword arg with default value `None`. The C++ side has `StreamOrDevice s = {}`; `nb::none()` maps to that default. |
| Raw string `R"(...)"` | Becomes `axpby.__doc__`. |

### Compiled output

The whole file compiles into one shared library file, e.g.
`_ext.cpython-312-darwin.so`. This sits next to `mlx_ext.metallib` (the compiled
Metal kernels) inside the installed package directory. After
`pip install -e .`, run `pip show -f mlx_sample_extensions` to see the layout.

### Why `mx::array` arguments Just Work

There's no explicit type-conversion code in `bindings.cpp` for `mx::array` ↔
`mlx.core.array`. Yet it works. The reason: **MLX has already taught nanobind**.

MLX's own bindings register `mx::array` with nanobind's type system, exposing
`mlx.core.array` as the Python counterpart. The CMake invocation tells our
extension's nanobind module to share that registry:

```cmake
# examples/extensions/CMakeLists.txt
nanobind_add_module(
  _ext
  ...
  NB_DOMAIN mlx              # ★ share type registrations with MLX's nanobind domain
  ...
```

`NB_DOMAIN mlx` is the magic word. Without it, the extension's `mx::array` would
be a separate type from MLX's perspective and conversion would fail. With it,
`mx::array` arguments and return values transparently convert to/from
`mlx.core.array` Python objects. Same for `Stream`, `Device`, dtype enums, etc.

**Implication for the SpQt extension:** we don't write any glue code for our
op's `mx::array` arguments. The same `NB_DOMAIN mlx` line in our CMake gets us
free type translation.

### The `__init__.py` file

```python
# mlx_sample_extensions/__init__.py  (5 lines)
import mlx.core as mx                # 1) ensure MLX is loaded first
from ._ext import axpby              # 2) re-export the C++ binding
```

Two purposes:

**1. Initialization order.** `import mlx.core as mx` ensures MLX is loaded *before*
our `_ext` module. MLX's nanobind type registrations live in `mlx.core`'s
compiled extension; those registrations must exist when `_ext` is loaded so its
`mx::array` translations resolve. Importing `_ext` before `mlx.core` can produce
obscure "type not registered" errors at call time.

**2. Public-API surface.** Without re-export, users would write
`from mlx_sample_extensions._ext import axpby` — the leading underscore signals
"internal". With the re-export, the clean form `from mlx_sample_extensions import
axpby` works. `_ext` is the *implementation detail*; `__init__.py` is the
*public API*.

`__init__.py` is also where Python-level wrappers go if needed. For axpby none
are needed, so it's just a re-export.

### Implications for the SpQt extension

| Concern | What we'll do |
|---|---|
| Module name | Rename `_ext` → `_spqt_ext` (private), `mlx_sample_extensions` → `mlx_spqt` (public). |
| C++ function bindings | One `m.def(...)` per registered op. M1: `zigzag_qmv_dense`. M2: `zigzag_qmv_sparse`. |
| `NB_DOMAIN mlx` | Keep verbatim — same magic gets us `mx::array` translation for free. |
| Python wrapper | `zigzag_quantize` is pure Python/MLX (operates on fp weights, calls `mx.quantize`, etc.) — lives in `mlx_spqt/__init__.py`, not in `bindings.cpp`. The C++ ops are exposed via `from ._spqt_ext import ...`. |
| Initialization order | Keep `import mlx.core as mx` at the top of `__init__.py`. Same reason. |

### Summary table

| Question | Answer |
|---|---|
| What is bindings.cpp's job? | Declare which C++ functions are Python-visible and how arguments translate. |
| What does NB_MODULE produce? | The C entry-point function for the Python module (`PyInit__ext`); generates the trampoline code at compile time. |
| How does `mx::array` translate? | Inherited from MLX via `NB_DOMAIN mlx` in the CMake invocation. No per-extension glue. |
| What's the role of `__init__.py`? | (a) Force MLX to load first so type registrations exist; (b) re-export the C++ binding under a clean public name. |
| Why `_ext` underscored? | Python convention for "internal" — public API is the package, not the compiled module. |

---

## Order 2 — `my_ext::axpby()` op function (`axpby/axpby.cpp:44-75`)

The C++-side counterpart to the Python `axpby()` callable. **Constructs a lazy
graph node and returns it; does no compute.** Same pattern every MLX op function
follows, including built-ins.

### Function signature

```cpp
mx::array axpby(
    const mx::array& x,           // input array x
    const mx::array& y,           // input array y
    const float alpha,            // scalar
    const float beta,             // scalar
    mx::StreamOrDevice s = {});   // optional, defaults to current default stream
```

- Arrays passed by `const&`: cheap, `mx::array` is a small wrapper around a `shared_ptr`.
- `mx::StreamOrDevice` is a variant accepting `Stream`, `Device`, or empty. Inside the function, `to_stream(s)` resolves to a concrete `Stream`.

### Four-step pattern

The body breaks into four conceptual steps:

#### Step 1 — Dtype promotion (lines 51-57)

```cpp
auto promoted_dtype = promote_types(x.dtype(), y.dtype());
auto out_dtype = mx::issubdtype(promoted_dtype, mx::float32)
    ? promoted_dtype
    : promote_types(promoted_dtype, mx::float32);
```

Two-stage:
1. NumPy-style binary promotion between x's and y's dtypes.
2. Force-upcast non-floating to float32 (axpby's kernel is only implemented for fp).

#### Step 2 — Insert casting primitives (lines 60-61)

```cpp
auto x_casted = mx::astype(x, out_dtype, s);
auto y_casted = mx::astype(y, out_dtype, s);
```

`mx::astype` is itself an op that **inserts an `AsType` primitive node into the
lazy graph if needed**. If the dtype already matches, it's a no-op (returns the
same array; no node added). Otherwise, the result is a new lazy array whose
primitive is `AsType` and whose input is the original. **The cast is lazy** —
runs when `mx::eval` fires, batched with everything else.

Passing `s` ensures the cast op lands on the caller's chosen stream.

#### Step 3 — Insert broadcast primitives (lines 64-65)

```cpp
auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
auto out_shape = broadcasted_inputs[0].shape();
```

Same lazy-insertion pattern: `broadcast_arrays` inserts `Broadcast` primitive
nodes if shapes differ; otherwise no-op. Output shape comes from any of the
post-broadcast inputs (all have the same shape after broadcasting).

#### Step 4 — Construct the result array (lines 69-74)

```cpp
return mx::array(
    /* shape =     */ out_shape,
    /* dtype =     */ out_dtype,
    /* primitive = */ std::make_shared<Axpby>(to_stream(s), alpha, beta),
    /* inputs =    */ broadcasted_inputs);
```

**The heart of the op function.** The 4-arg `mx::array` constructor:

```cpp
mx::array::array(Shape, Dtype, std::shared_ptr<Primitive>, std::vector<array>);
```

What this constructor actually does (per `mlx/array.h`):
1. Allocates an `array_desc_` (heap-allocated descriptor).
2. Sets shape, dtype.
3. Sets `primitive` shared_ptr.
4. Sets `inputs` vector.
5. `status = Status::unscheduled`.
6. **Does NOT allocate buffer memory.** The `data` field is empty.

The returned array is a **lazy graph node**: knows the op (`Axpby`), the
dependencies (the broadcasted inputs), the eventual shape/dtype — but no values
yet. They materialize only when `mx::eval` runs.

The `std::make_shared<Axpby>(to_stream(s), alpha, beta)` carries the op's
*parameters* (alpha, beta, stream). When `eval_gpu` later fires, those member
values get baked into the kernel arguments at dispatch.

### What the lazy graph looks like after the call

```
                    out ← Axpby(α=2.0, β=3.0)
                          /        \
                  x_broadcast    y_broadcast      ← only if shapes differ
                       │              │
                  x_casted        y_casted        ← only if dtypes differ
                       │              │
                       x              y           ← the user's leaves
```

Each box is an `mx::array` with a primitive pointer and an inputs list. None
have data yet. `mx::eval(out)` triggers the DFS → tape → dispatch from
`mlx-execution-model.md` §3.

### The same pattern in MLX core

`mlx/ops.cpp:4477-4524` (paraphrased):

```cpp
array quantized_matmul(...) {
  // ...validation, shape extraction...
  return array(
    out_shape,
    x.dtype(),
    std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits, transpose, qmode),
    {x, w, scales, biases});
}
```

Identical structure. **Whether an op is built into MLX core or comes from an
extension, the lazy-graph integration uses the same constructor and produces
the same kind of node.** That's what makes extension ops indistinguishable from
core ops at eval time.

### Implications for the SpQt extension

For `mlx_spqt::zigzag_qmv_sparse(x, w_zz, scales, biases, idx, s)`:

| axpby step | What we'll do |
|---|---|
| Dtype promotion | **Skip.** MVP hardcodes fp16. Validate `x.dtype() == float16` and throw otherwise. |
| Casting | **Skip.** No casts needed; MVP enforces fp16. |
| Broadcasting | **Skip.** Single fixed shape; no broadcast. |
| Output shape | Compute directly: `out_shape = {x.shape(0), n_out}`. One line. |
| Primitive construction | `std::make_shared<ZigzagQMVSparse>(to_stream(s), group_size, bits)` carrying our parameters. |
| Array constructor | Same 4-arg call: `mx::array(out_shape, mx::float16, primitive, {x, w_zz, scales, biases, idx})`. |

Net: our op function is shorter than axpby's (~10-20 lines vs. ~30) because of
MVP scoping. If we ever generalized, we'd grow back toward axpby's pattern.

### Summary table

| Question | Answer |
|---|---|
| What is the op function's job? | Construct a lazy graph node — validate inputs, prepare them (cast/broadcast), build a Primitive, return an `mx::array` referencing both. |
| Where does compute happen? | **Not here.** Only when something downstream triggers `mx::eval`. |
| What is `mx::astype` / `broadcast_arrays`? | Themselves ops that insert primitive nodes into the lazy graph. No-ops if no work needed. |
| What is `to_stream(s)` for? | Resolves the `StreamOrDevice` variant to a concrete `Stream` (default-stream-of-default-device if unset). The Primitive needs a real `Stream`. |
| Why pass `s` to `astype` and `broadcast_arrays`? | So those inserted ops land on the same stream as the main op — matters for cross-stream sync. |
| What are the Primitive's member fields for? | They're the op's parameters (alpha, beta for axpby; group_size, bits for SpQt). Captured at construction; consumed by `eval_gpu` at dispatch. |

---

## Order 3 — `Axpby` Primitive class declaration (`axpby/axpby.h`)

The class definition. ~70 lines of substantive declaration. Plugs into MLX's
runtime by inheriting from `mx::Primitive` and overriding the right virtual
methods.

### The base class

`mx::Primitive` (in `mlx/primitives.h`) is the abstract base for **every op in
MLX** — `Add`, `Multiply`, `QuantizedMatmul`, `Softmax` all inherit from it.
The base class:

- Stores a `Stream` (runtime knows what stream to dispatch on).
- Declares pure-virtual `eval_cpu` / `eval_gpu` that subclasses must implement.
- Declares overridable `jvp` / `vjp` / `vmap` (default: throw NYI).
- Provides `stream()`, `name()`, `is_equivalent()` hooks.

By inheriting from `Primitive`, `Axpby` plugs into MLX's lazy-eval pipeline:
the lazy graph holds `shared_ptr<Primitive>`; the eval pipeline calls the
right virtual method based on the subclass it actually is.

### Constructor — captures parameters

```cpp
explicit Axpby(mx::Stream stream, float alpha, float beta)
    : mx::Primitive(stream), alpha_(alpha), beta_(beta) {};
```

- **`explicit`** prevents implicit conversions (defensive C++ practice).
- **`mx::Primitive(stream)`** calls the base-class constructor. The stream
  is stored in the base for later access via `primitive.stream()`.
- **`alpha_(alpha), beta_(beta)`** stores the op's parameters.

**This is the primary purpose of a Primitive subclass: capture the op's
parameters at construction.** Empty body means there's no extra logic.

When the op function calls
`std::make_shared<Axpby>(to_stream(s), alpha, beta)`, this is what runs. The
created instance lives in the returned `mx::array`'s
`array_desc_->primitive`. When eval fires later, the runtime invokes
`eval_gpu` on this instance, which reads `alpha_` and `beta_` via `this->`.

### Method roster

The declarations at a glance:

| Method | Purpose | What we'll do for SpQt MVP |
|---|---|---|
| `eval_cpu` | CPU compute dispatch | Throw NYI (no CPU path) |
| `eval_gpu` | GPU compute dispatch — kernel encoding | **Implement** (the real work) |
| `jvp` | Forward-mode autograd | Throw NYI (out of scope) |
| `vjp` | Reverse-mode autograd | Throw NYI (out of scope) |
| `vmap` | Vectorization (batch axis) | Throw NYI (out of scope) |
| `name()` | Debug/repr identifier | Return `"ZigzagQMVSparse"` |
| `is_equivalent` | Graph dedup under `mx.compile` | Compare member fields |

#### `eval_cpu` / `eval_gpu`

```cpp
void eval_cpu(const std::vector<mx::array>& inputs,
              std::vector<mx::array>& outputs) override;
void eval_gpu(const std::vector<mx::array>& inputs,
              std::vector<mx::array>& outputs) override;
```

Two things to flag:

- **`override` keyword** (C++11). Tells the compiler "I'm overriding a virtual
  method." Compiler errors if the signature doesn't match an actual virtual
  method (typo, missing `const`, etc.). Catches subtle bugs.
- **`outputs` is a vector** — for ops with multiple outputs (e.g. `eigh`
  returns eigenvalues + eigenvectors). Axpby has one, so `outputs[0]` is the
  populated cell.

#### `jvp` — forward-mode autograd

```cpp
std::vector<mx::array> jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) override;
```

**Jacobian-vector product**: forward-mode autodiff. Computes `J·v` where `v`
is a tangent direction.

Math for axpby (`z = αx + βy`): `∂z/∂x = α`, `∂z/∂y = β`. The implementation
at `axpby.cpp:253-275` recognizes that `jvp` of `axpby` is itself an `axpby`:

```cpp
return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
```

Beautifully recursive. `argnums` lists which input indices we're
differentiating against (allows partial gradients).

#### `vjp` — reverse-mode autograd

```cpp
std::vector<mx::array> vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) override;
```

**Vector-Jacobian product**: reverse-mode (the standard backprop). Computes
`v^T·J` where `v` is the upstream cotangent (gradient of loss wrt output).

For axpby: `∂L/∂x = α * ∂L/∂z`, `∂L/∂y = β * ∂L/∂z`.

`QuantizedMatmul::vjp` in MLX core (`mlx/primitives.cpp:3459-3525`) explicitly
throws "no gradient wrt the quantized weights" — quantization isn't
differentiable in the usual sense. Same precedent for our SpQt VJP.

#### `vmap` — vectorization

```cpp
std::pair<std::vector<mx::array>, std::vector<int>> vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) override;
```

When the user calls `mx.vmap(my_fn)(batched_x)`, MLX asks each primitive how
to handle a batch axis. Returns `(output arrays, list of axes per output)`.

Axpby itself throws NYI (`axpby.cpp:294-298`); vmap support is optional. Our
SpQt MVP throws too.

#### `name()` — debugging

```cpp
const char* name() const override { return "Axpby"; }
```

String identifier. Used by:
- `array.__repr__()` for graph visualization.
- Error messages.
- Debug trace tools.

Convention: same as the class name. For SpQt: `"ZigzagQMVSparse"`.

#### `is_equivalent` — graph dedup

```cpp
bool is_equivalent(const mx::Primitive& other) const override;
```

Implementation at `axpby.cpp:301-304`:

```cpp
bool Axpby::is_equivalent(const Primitive& other) const {
  const Axpby& r_other = static_cast<const Axpby&>(other);
  return alpha_ == r_other.alpha_ && beta_ == r_other.beta_;
}
```

**Used by `mx.compile`** to identify equivalent primitives and dedupe in the
compiled graph. Two `Axpby`s are equivalent if their parameters match.

- Doesn't check input-array identity (those are tracked separately).
- `static_cast` is safe because the runtime has already verified `other` is
  the same subclass type before calling.

For SpQt: compare `group_size_`, `bits_`, and any tuning knobs we expose.

### Private member fields

```cpp
private:
  float alpha_;
  float beta_;
```

Captured by constructor, read by `eval_gpu`/`eval_cpu`/`vjp`/`jvp`, checked
by `is_equivalent`.

**These are the op's "configuration" — anything the kernel needs that isn't in
the input tensors.** Trailing underscore is the MLX (and Google C++)
convention for member fields.

For SpQt: `group_size_`, `bits_` — possibly tuning knobs (`TGs_per_band_`,
`n_simdgroups_`) if we expose them.

### Header / implementation split

`.h` declares the interface; `.cpp` has implementations. Standard C++ pattern,
but worth noting:

- **`.h`** is included anywhere the class is *used* — `bindings.cpp` includes
  `axpby.h` to know about `my_ext::axpby` and `Axpby`.
- **`.cpp`** has implementations, compiled once.
- **Why not inline in the header?** Implementations are large and have many
  dependencies (e.g. `eval_gpu` includes Metal headers). Keeping them in `.cpp`
  bounds the per-translation-unit compile cost and keeps the header
  dependency-light.

### What our SpQt M0b will write

Per Working Principle #1 (minimal code) and our scope decisions:

```cpp
// extensions/mlx_spqt/spqt/spqt.h  (sketch — actual implementation in M2)
class ZigzagQMVSparse : public mx::Primitive {
 public:
  explicit ZigzagQMVSparse(mx::Stream stream, int group_size, int bits)
      : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};

  void eval_cpu(const std::vector<mx::array>&,
                std::vector<mx::array>&) override {
    throw std::runtime_error("ZigzagQMVSparse: no CPU implementation");
  }
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;   // in .cpp

  std::vector<mx::array> jvp(...) override {
    throw std::runtime_error("ZigzagQMVSparse: no JVP");
  }
  std::vector<mx::array> vjp(...) override {
    throw std::runtime_error("ZigzagQMVSparse: no VJP");
  }
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(...) override {
    throw std::runtime_error("ZigzagQMVSparse: no vmap");
  }

  const char* name() const override { return "ZigzagQMVSparse"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const ZigzagQMVSparse&>(other);
    return group_size_ == o.group_size_ && bits_ == o.bits_;
  }

 private:
  int group_size_;
  int bits_;
};
```

~30 lines of declarations. Scopes match what the MVP actually needs:
GPU-only forward, no autograd, single configuration. Throws are explicit
"out of scope, by design", not gaps to fill in later.

### Summary

| Question | Answer |
|---|---|
| What is the Primitive class for? | Capture the op's parameters and plug into MLX's runtime via virtual-method overrides. |
| What's required for a working op? | `eval_cpu` or `eval_gpu` (depending on backend), `name()`, `is_equivalent()`. The rest can throw NYI. |
| What do the member fields hold? | The op's configuration — anything the kernel needs that isn't in the input tensors. |
| When does the constructor run? | When the op function calls `std::make_shared<Axpby>(...)`. |
| When do the virtual methods run? | `eval_*` at lazy-graph eval time; `jvp`/`vjp`/`vmap` only if user calls `mx.grad`/`mx.vjp`/`mx.vmap`; `is_equivalent` only under `mx.compile`. |
| Why throw NYI for autograd? | Quantization isn't differentiable in the usual sense; we're not training; same precedent as `QuantizedMatmul::jvp` in MLX core. |

---

## Order 4 — `Axpby::eval_gpu` (`axpby/axpby.cpp:153-235`)

The actual GPU dispatch: encoder setup, kernel lookup, buffer binding, dispatch
encoding. **The closest analog to what M2 will write for SpQt's
`ZigzagQMVSparse::eval_gpu`.** Worth understanding line by line.

### The wrapper-layer detour: why no `setBuffer` calls?

A common first-read confusion: where are the Metal `setBuffer:offset:atIndex:`
calls? They're *inside* MLX's wrapper. MLX wraps `MTLComputeCommandEncoder` in
its own `metal::CommandEncoder` class
(`mlx/backend/metal/device.h:84-180`). The wrapper exposes:

| Wrapper call | Underlying Metal call | Used for |
|---|---|---|
| `set_input_array(a, idx)` | `setBuffer:offset:atIndex:idx` | bind `mx::array` as input (read dependency) |
| `set_output_array(a, idx)` | `setBuffer:offset:atIndex:idx` | bind `mx::array` as output (write target) |
| `set_bytes(value, idx)` | `setBytes:length:atIndex:idx` | inline copy a small POD scalar |
| `set_vector_bytes(vec, idx)` | `setBytes:length:atIndex:idx` | inline copy a vector's contents |

The wrapper does five things the raw Metal API doesn't:

1. **Extracts `MTLBuffer*`** from `mx::array.data<T>()` (with the right offset for views).
2. **Tracks input/output dependencies** so the completion handler keeps input buffers alive until the GPU is done reading them.
3. **Manages residency** via `MTLResidencySet` so kernels don't crash on swapped-out buffers.
4. **Updates per-buffer accounting** (`buffer_ops_`, `buffer_sizes_`) that drives `needs_commit()` (per `mlx-execution-model.md` §4).
5. **Picks `setBuffer` vs `setBytes`** based on whether the binding is an array or a scalar.

### Buffer slots — the C++/MSL connection

The `idx` argument (0, 1, 2, ...) is the **buffer slot index** that must match
the `[[buffer(N)]]` attribute in the MSL kernel signature. From axpby.metal:

```metal
[[kernel]] void axpby_general(
    device const T* x        [[buffer(0)]],   // ← matches set_input_array(x, 0)
    device const T* y        [[buffer(1)]],   // ← matches set_input_array(y, 1)
    device T* out            [[buffer(2)]],   // ← matches set_output_array(out, 2)
    constant const float& alpha [[buffer(3)]],// ← matches set_bytes(alpha_, 3)
    constant const float& beta  [[buffer(4)]],
    constant const int* shape   [[buffer(5)]],
    constant const int64_t* x_strides [[buffer(6)]],
    constant const int64_t* y_strides [[buffer(7)]],
    constant const int& ndim          [[buffer(8)]],
    uint index [[thread_position_in_grid]]) { ... }
```

The indices line up exactly. **A common bug pattern**: edit the kernel
signature without updating the `set_*` calls (or vice versa) — buffers go to
the wrong slots, kernel reads garbage, no clean error.

### `device` vs `constant` qualifiers

Notice in the MSL signature:
- `device const T* x` — pointer to GPU **device memory** (a real `MTLBuffer`).
- `constant const float& alpha` — pointer/reference to **constant memory**
  (small, read-only, fast cache).

Metal routes them automatically based on how the C++ side bound them:
- `setBuffer:` → device memory.
- `setBytes:` → constant memory (bytes copied into a per-dispatch constant region).

Use `set_bytes` for small scalars + small metadata vectors (typically ≤ 4 KB
total per dispatch); `set_input_array` / `set_output_array` for large tensors.

### The eight-step structure of eval_gpu

```cpp
void Axpby::eval_gpu(const std::vector<mx::array>& inputs,
                    std::vector<mx::array>& outputs) {
  // 1. Get inputs/output references
  auto& x = inputs[0];
  auto& y = inputs[1];
  auto& out = outputs[0];

  // 2. Get stream + Metal device handle
  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  // 3. Choose a kernel specialization based on input contiguity
  bool contiguous_kernel = (x.flags().row_contiguous && y.flags().row_contiguous) ||
                           (x.flags().col_contiguous && y.flags().col_contiguous);

  // 4. Allocate output buffer
  if (contiguous_kernel) {
    out.set_data(mx::allocator::malloc(...), x.data_size(), x.strides(), x.flags());
  } else {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }

  // 5. Build the kernel name and look it up in the metallib
  std::string kname = "axpby_";
  kname += (contiguous_kernel ? "contiguous_" : "general_");
  kname += type_to_name(out);                            // → "axpby_contiguous_float32"
  auto lib = d.get_library("mlx_ext", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);

  // 6. Get/create the encoder for this stream and bind the kernel pipeline
  auto& compute_encoder = mx::metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  // 7. Bind inputs/output/scalars to buffer slots
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(y, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(alpha_, 3);
  compute_encoder.set_bytes(beta_, 4);
  if (!contiguous_kernel) { /* shape/strides/ndim */ }

  // 8. Decide threadgroup geometry and dispatch
  size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());
  MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(nelem, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}
```

Three logical groups:

| Group | Steps | Purpose |
|---|---|---|
| **Setup** | 1-2 | Inputs, output, stream, MTLDevice |
| **Kernel resolution** | 3-6 | Pick variant; look up by name from metallib; bind pipeline |
| **Dispatch** | 7-8 | Bind buffers + scalars; choose threadgroup geometry; encode |

**The encoder does not run the kernel.** It adds a dispatch command to the
current command buffer. Actual GPU execution happens at commit time
(`needs_commit()` boundary or `gpu::finalize()` at end of eval).

### Step 4 in detail — output allocation

```cpp
out.set_data(mx::allocator::malloc(out.nbytes()));
```

Until this line, `out` is unmaterialized — descriptor exists, buffer field
empty. `set_data` allocates GPU memory and stores the `MTLBuffer` in `out`'s
descriptor. This is why `set_output_array(out, 2)` later can extract a real
buffer.

For SpQt our eval_gpu will do the same: `out.set_data(mx::allocator::malloc(out.nbytes()))` before binding `out` as an output. The SpQt subtlety: if we use atomic-add for cross-TG reduce, we may want to pre-zero `out` (an extra small kernel
or a `set_data` variant that zeros).

### Step 5 in detail — kernel name resolution

```cpp
std::string kname = "axpby_";
kname += (contiguous_kernel ? "contiguous_" : "general_");
kname += type_to_name(out);                            // e.g. "float32"
auto lib = d.get_library("mlx_ext", current_binary_dir());
auto kernel = d.get_kernel(kname, lib);
```

`d.get_library("mlx_ext", current_binary_dir())` loads the `.metallib` file
sitting next to the extension's `.so` (the runtime-path-resolution piece is
Order 6). `d.get_kernel(kname, lib)` looks up a kernel by name within the
library and returns an `MTLComputePipelineState`.

The kernel name must match a `[[host_name(...)]]` template instantiation in
the `.metal` file. axpby.metal has:

```cpp
instantiate_axpby(float32, float);   // produces host name "axpby_contiguous_float32" etc.
```

If the name doesn't resolve at runtime, you get a clear error
("Kernel not found"). Always-failing-clear is good — silent buffer-slot bugs
are worse.

### What SpQt's eval_gpu will look like

Same 8 steps, simpler in places because of MVP scope:

| Step | axpby | SpQt M2 |
|---|---|---|
| 1. Inputs | x, y (2) | x, w_zz, scales, biases, idx (5) |
| 2. Stream/device | identical | identical |
| 3. Variant pick | contiguity branch | one fixed variant (single shape/dtype) |
| 4. Output alloc | per-contiguity | always `out.nbytes()`; consider pre-zero for atomics |
| 5. Kernel name | `axpby_{contig\|general}_{dtype}` | one fixed name, e.g. `spqt_zigzag_qmv_sparse_half_gs64_b4` |
| 6. Pipeline bind | identical | identical |
| 7. Buffer binding | 5-9 calls | ~7 calls (5 arrays + `idx_count` + maybe N) |
| 8. Threadgroup geometry | linear over N | 2D over (`group_size` row bands × K-index slices) |

Total length: ~80 lines vs. axpby's ~85.

### Summary

| Question | Answer |
|---|---|
| Why no `setBuffer:` in eval_gpu? | MLX wraps `MTLComputeCommandEncoder`; the wrapper calls `setBuffer:` internally and adds dependency tracking, residency, accounting on top. |
| What does the slot index do? | Connects the C++ binding to the `[[buffer(N)]]` attribute in the MSL kernel signature. They must match. |
| `device` vs `constant` in MSL — who decides? | The C++ side: `set_input_array`/`set_output_array` → `device` (real MTLBuffer); `set_bytes`/`set_vector_bytes` → `constant` (inline-copied bytes). |
| When does the GPU actually run? | After encoding, when the command buffer is committed (per the execution-model dive). eval_gpu is encoding, not running. |
| What's the most fragile part? | Kernel name resolution (step 5). A typo or stale rename produces a clean runtime error; buffer-slot mismatches (step 7) produce silent corruption — keep the C++ and MSL signatures in sync. |

---

## Order 5 — `axpby.metal` (kernel templates + instantiation macros)

The Metal source. ~47 lines. Two kernel templates (contiguous + general), four
dtypes, instantiated via a small macro into 8 host-named symbols.

### Two kernels, one operation, two layout strategies

Both compute `out[i] = α*x[i] + β*y[i]` element-wise. They differ in **how they
read `x[i]` and `y[i]`** — i.e., how they handle memory layout.

#### `axpby_contiguous` (lines 25-35)

```metal
out[index] = static_cast<T>(alpha) * x[index] + static_cast<T>(beta) * y[index];
```

**Direct linear indexing.** Thread N reads `x[N]`, `y[N]`, writes `out[N]`.
Works only when `x`, `y`, `out` are all contiguous in memory with matching
strides — the "happy path" where the linear thread index *is* the memory
offset.

#### `axpby_general` (lines 7-23)

```metal
auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
auto y_offset = elem_to_loc(index, shape, y_strides, ndim);
out[index] = static_cast<T>(alpha) * x[x_offset] + static_cast<T>(beta) * y[y_offset];
```

**Stride-aware indexing.** Thread N reads `x[x_offset]` where `x_offset` is
computed from the logical thread coordinate via the shape and stride arrays.
Works on *any* layout — broadcasts, transposes, slices, non-contiguous views.

Cost of generality: per-thread `elem_to_loc` math, more buffer slots passed in
(shape + 2 stride arrays + ndim), more constant-memory pressure. Pure overhead
if data is contiguous.

### `elem_to_loc` — what it actually does

MLX-provided, in `mlx/backend/metal/kernels/utils.h`. Roughly:

```cpp
size_t elem_to_loc(uint elem, constant const int* shape,
                   constant const int64_t* strides, int ndim) {
    size_t offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        offset += (elem % shape[d]) * strides[d];
        elem /= shape[d];
    }
    return offset;
}
```

Converts a **linear logical index** to a **physical memory offset** given the
tensor's shape and strides. For a `(3, 4)` tensor:

| Layout | shape | strides | `elem_to_loc(5, ...)` |
|---|---|---|---|
| Contiguous (row-major) | `[3, 4]` | `[4, 1]` | `(5%4)*1 + (5/4%3)*4 = 5` (= linear index ✓) |
| Transposed view of `(4,3)` shape-reported as `(3,4)` | `[3, 4]` | `[1, 4]` | `(5%4)*4 + (5/4%3)*1 = 17` (≠ 5 — reflects the transpose) |

For contiguous layouts, the offset *equals* the linear index. That's why the
contiguous kernel can skip this computation.

### Why two static templates instead of one runtime branch

Metal can't dynamically decide layout-handling at runtime — the choice has to
be baked into the kernel. eval_gpu picks at C++ time:

```cpp
bool contiguous_kernel = (x.flags().row_contiguous && y.flags().row_contiguous) ||
                         (x.flags().col_contiguous && y.flags().col_contiguous);
```

If true, dispatch the contiguous variant; pass fewer buffers (skip shape/strides/ndim).
If false, dispatch the general variant with full layout info.

### The instantiation macro chain

```cpp
#define instantiate_axpby(type_name, type)                             \
  instantiate_kernel("axpby_general_" #type_name, axpby_general, type) \
  instantiate_kernel("axpby_contiguous_" #type_name, axpby_contiguous, type)

instantiate_axpby(float32, float);
instantiate_axpby(float16, half);
instantiate_axpby(bfloat16, bfloat16_t);
instantiate_axpby(complex64, complex64_t);
```

Two macros, two layers:

#### `instantiate_axpby` — local convenience

Stringifies `type_name` into the symbol name and pairs each call with two
`instantiate_kernel` invocations (one per template). Pure convenience; could
be unrolled.

#### `instantiate_kernel` — MLX's machinery

Defined in MLX kernel utils. Roughly expands to:

```metal
template [[host_name("axpby_general_float32")]]
[[kernel]] void axpby_general<float>(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    /* ...args with T resolved to float... */);
```

The `[[host_name("...")]]` attribute is what makes the symbol name
**discoverable from the host C++ side** via `d.get_kernel("axpby_contiguous_float32", lib)`.
Without `host_name`, the Metal compiler would mangle the template into something
unreadable.

After full expansion the macro tree produces **8 host-named symbols** in the
metallib:

```
axpby_general_float32      axpby_contiguous_float32
axpby_general_float16      axpby_contiguous_float16
axpby_general_bfloat16     axpby_contiguous_bfloat16
axpby_general_complex64    axpby_contiguous_complex64
```

These are the symbols `d.get_kernel(name, lib)` looks up at runtime.

### Connection back to eval_gpu

eval_gpu's name string:

```cpp
std::string kname = "axpby_";
kname += (contiguous_kernel ? "contiguous_" : "general_");
kname += type_to_name(out);   // "float32"
// kname == "axpby_contiguous_float32"
```

…matches one of the 8 symbols exactly. **The instantiation list and eval_gpu's
name-construction must stay in sync** — if you add a kernel name in the metal
file, eval_gpu has to know how to construct it; if you add a dtype path in
eval_gpu, the metal file has to instantiate it. Kernel-side analog of the
buffer-slot consistency rule from Order 4.

### What we'll write for SpQt

More substance per kernel (zigzag walk + simdgroup reduce + atomic-add) but
fewer kernels (single fixed configuration for the MVP).

```metal
// extensions/mlx_spqt/spqt/spqt.metal  (sketch — actual implementation in M2)
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/quantized.h"   // for qdot, load_vector (Working Principle #4)

template <typename T, int group_size, int bits>
[[kernel]] void zigzag_qmv_sparse_impl(
    device const uint32_t* w_zz       [[buffer(0)]],
    device const T*        scales_zz  [[buffer(1)]],
    device const T*        biases_zz  [[buffer(2)]],
    device const T*        x          [[buffer(3)]],
    device const int32_t*  idx        [[buffer(4)]],
    constant const int&    idx_count  [[buffer(5)]],
    constant const int&    K          [[buffer(6)]],
    device atomic_float*   out        [[buffer(7)]],   // atomic for cross-TG reduce
    uint3 tid              [[threadgroup_position_in_grid]],
    uint sg_id             [[simdgroup_index_in_threadgroup]],
    uint sg_lane           [[thread_index_in_simdgroup]]) {

  // Compile-time constexprs derived from <T, group_size, bits> per qmv_fast pattern
  constexpr int pack_factor      = get_pack_factor<bits, 32>();
  constexpr int values_per_thread = pack_factor * (bits == 2 ? 1 : 2);
  // ...

  // ... idx-driven K-walk, qdot<>(...) inner unpack, simd_sum reduce,
  //     atomic_fetch_add_explicit on out[row] ...
}

// Single MVP instantiation
template [[host_name("zigzag_qmv_sparse_half_gs64_b4_M64")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 64>(
    device const uint32_t*, device const half*, device const half*,
    device const half*, device const int32_t*,
    constant const int&, constant const int&,
    device atomic_float*, uint3, uint, uint);
```

For M1 (dense variant): identical signature minus `idx`/`idx_count`, contiguous
K-walk instead of idx-driven. Two kernels total, one
`template [[host_name(...)]]` per kernel — no macro needed for a single
specialization. If we generalize to multi-bit or multi-group-size later, *then*
introduce a macro à la `instantiate_axpby`.

### One detail worth flagging — `atomic_float*`

Note `out` is typed `device atomic_float*` rather than `device float*`. Required
for `atomic_fetch_add_explicit` to compile. **This is one of the M0b smoke
tests** — because if the toolchain doesn't accept `atomic_float*` outputs
bound from `mx.array` of dtype float, we need a workaround (e.g., one-TG-per-row
to avoid the cross-TG atomic).

### Summary

| Question | Answer |
|---|---|
| Why two kernel templates? | Static decision at Metal-compile time about whether to compute strided offsets per thread. Contiguous skips it; general handles arbitrary layouts. |
| What is `elem_to_loc`? | MLX-provided helper that converts a linear thread index to a physical memory offset given shape+strides. Free for contiguous data; per-thread cost otherwise. |
| What does `[[host_name(...)]]` do? | Bridges the Metal symbol table to `d.get_kernel(...)`. Without it, the template instantiation has a mangled name not findable from C++. |
| What's the consistency rule? | The instantiation list (this file) and eval_gpu's name-construction (axpby.cpp) must stay aligned. Adding a name in one without the other → "kernel not found" or silent slot-misalignment. |
| What macro pattern do we use for SpQt MVP? | None — one explicit `template [[host_name(...)]]` per kernel. Macro is overkill for a single specialization; introduce only if we generalize across bits/group_size. |

---

## Order 6 — `current_binary_dir()` (`axpby/axpby.cpp:20-31`)

A 12-line helper that finds the directory containing the loaded extension `.so`,
so the runtime can locate the sibling `.metallib`. Short but worth understanding —
this is what makes the extension portable across install paths.

### The problem it solves

When `eval_gpu` calls `d.get_library("mlx_ext", binary_dir)`, MLX must find
`mlx_ext.metallib` on disk. The metallib lives **next to the extension's `.so`**
in the installed package directory. But pip can install the package anywhere
(conda's site-packages, user pip's site-packages, virtualenv, editable install,
…). Hardcoding paths fails; CWD fails. The kernel needs to find its sibling
metallib regardless of where the package landed.

### The solution

```cpp
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
```

Three layers:

#### Layer 1 — `dladdr` (POSIX)

`dladdr(addr, &info)` from `<dlfcn.h>` asks the dynamic linker: "given this
address, which loaded shared library does it belong to?" Fills `Dl_info`:

```cpp
struct Dl_info {
    const char* dli_fname;   // file path of the shared library  ← we want this
    void*       dli_fbase;   // base address of the .so
    const char* dli_sname;   // nearest exported symbol name
    void*       dli_saddr;   // nearest exported symbol address
};
```

#### Layer 2 — the bootstrap trick

```cpp
dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)
```

The address is `&current_binary_dir` — a function pointer to *this very
function*. Since `current_binary_dir` is defined inside the extension's `.so`,
its address must lie within the loaded library's address range. `dladdr`
resolves the address to the library, and `dli_fname` is its disk path.

`reinterpret_cast<void*>` is required because C++ doesn't implicitly convert
function pointers to data pointers (different sizes on some ABIs; identical on
macOS/Linux in practice).

#### Layer 3 — strip filename, keep directory

```cpp
std::filesystem::path(info.dli_fname).parent_path().string();
```

Turns:
```
/Users/.../site-packages/mlx_sample_extensions/_ext.cpython-312-darwin.so
```
into:
```
/Users/.../site-packages/mlx_sample_extensions
```

That's the directory where the metallib lives.

### The static-lambda caching idiom

```cpp
static std::string binary_dir = []() { /* compute */ }();
```

- Lambda creates a temporary callable; trailing `()` invokes it.
- `static std::string binary_dir = ...` stores the result in a function-local static.
- Per C++11 thread-safe statics: **initialization runs exactly once on first call**, with thread-safe synchronization. Subsequent calls return the cached value.

So `dladdr` runs once per process; every subsequent `current_binary_dir()` is a
near-free string copy.

### Why the metallib is there in the first place

The whole mechanism works because `_ext.so` and `mlx_ext.metallib` are
**co-installed** in the same directory. Two pieces ensure that:

#### CMakeLists — building the metallib next to the `.so`

`examples/extensions/CMakeLists.txt:46-58`:

```cmake
mlx_build_metallib(
    TARGET mlx_ext_metallib
    TITLE  mlx_ext                                  # → filename: mlx_ext.metallib
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
```

`mlx_build_metallib` is an MLX-provided macro: runs `xcrun metal` (compile each
`.metal`) and `xcrun metallib` (archive into a `.metallib`). Output goes in
`CMAKE_LIBRARY_OUTPUT_DIRECTORY` — the same directory CMake puts the `_ext.so`.

#### setup.py — installing the metallib alongside

`examples/extensions/setup.py:15`:

```python
package_data={"mlx_sample_extensions": ["*.so", "*.dylib", "*.metallib"]},
```

Tells setuptools to include `.metallib` files when packaging. After
`pip install`, the metallib gets copied into the same install directory as the
`.so`.

The chain: CMake builds them adjacent → setup.py packages them together → pip
installs them adjacent → `dladdr` finds the `.so` → `parent_path` finds the
directory → metallib loads from sibling.

### What we'll write for SpQt

**Verbatim** — copy the function as-is, change one identifier:

```cpp
// extensions/mlx_spqt/spqt/spqt.cpp  (helper at top of file)
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
```

In the `eval_gpu` call, the metallib name changes:

```cpp
auto lib = d.get_library("mlx_spqt", current_binary_dir());
//                       ^^^^^^^^^^ matches TITLE in our CMakeLists' mlx_build_metallib
```

The `TITLE` in our CMakeLists must match — we set it to `mlx_spqt`. That's the
only coupling worth tracking.

### Failure modes worth knowing

| Symptom | Cause | Fix |
|---|---|---|
| `Unable to get current binary dir.` thrown | `dladdr` returned 0. Rare on macOS/Linux unless the function was inlined or the binary stripped. | Don't inline `current_binary_dir`. The static-lambda pattern naturally prevents inlining. |
| `Failed to load library mlx_ext.metallib` from `eval_gpu` | The `.metallib` isn't next to the `.so`. Usually `package_data` missing `*.metallib`, or CMake's `OUTPUT_DIRECTORY` placed the metallib elsewhere. | Verify post-install: `pip show -f <package>` shows both files in the same dir. |
| Path returned has `build/lib.../` (editable install) | The `.so` ended up in a build dir rather than the package dir. | Verify with `python -c "import _ext; print(_ext.__file__)"`; fix CMake's `OUTPUT_DIRECTORY` if needed. |
| Wrong (old) metallib loaded | Stale build artifacts from before a kernel-source edit or rename. | `rm -rf build/ && pip install -e .` for a clean rebuild. |

### Summary

| Question | Answer |
|---|---|
| What does `current_binary_dir()` find? | The directory of the extension's `.so`, regardless of where pip installed it. |
| Why is `&current_binary_dir` passed to `dladdr`? | It's an address inside *this* `.so`. The dynamic linker resolves it back to that library's path. |
| Why static-lambda? | Compute once on first call, cache forever; thread-safe via C++11 static initialization. |
| Why is the metallib next to the `.so`? | `mlx_build_metallib` puts it in `CMAKE_LIBRARY_OUTPUT_DIRECTORY`; setup.py's `package_data` includes `*.metallib` so pip co-installs them. |
| What changes for SpQt? | Copy verbatim. Update the metallib name (`mlx_ext` → `mlx_spqt`) in `d.get_library(...)` and in CMakeLists' `TITLE`. |

---

## Order 7 — CPU implementation + autograd hooks (`axpby/axpby.cpp:80-144` and `:249-298`)

Both groups are **throw-NYI territory** for SpQt MVP — but worth seeing the
patterns at least once, both for completeness and so the talk can frame what
we deferred.

### Part A — CPU implementation (`:80-144`)

The CPU side mirrors eval_gpu's structure, but uses MLX's CPU encoder + lambda
dispatch instead of Metal kernel dispatch.

#### `axpby_impl<T>` — the templated worker

```cpp
template <typename T>
void axpby_impl(const mx::array& x, const mx::array& y, mx::array& out,
                float alpha_, float beta_, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(y);
  encoder.set_output_array(out);

  encoder.dispatch([x_ptr = x.data<T>(), y_ptr = y.data<T>(),
                    out_ptr = out.data<T>(),
                    size = out.size(), shape = out.shape(),
                    x_strides = x.strides(), y_strides = y.strides(),
                    alpha_, beta_]() {
    T alpha = static_cast<T>(alpha_);
    T beta = static_cast<T>(beta_);
    for (size_t out_idx = 0; out_idx < size; out_idx++) {
      auto x_offset = mx::elem_to_loc(out_idx, shape, x_strides);
      auto y_offset = mx::elem_to_loc(out_idx, shape, y_strides);
      out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
    }
  });
}
```

Three things to notice:

**1. Same wrapper pattern as Metal.** `mx::cpu::get_command_encoder(stream)`
returns a CPU-side `CommandEncoder`. Same `set_input_array` /
`set_output_array` API as the Metal encoder — MLX deliberately keeps the
encoder API symmetric across backends.

**2. `encoder.dispatch(lambda)` instead of kernel dispatch.** The CPU encoder
takes a *lambda* carrying the work. Captures all data it needs *by value* (raw
pointers from `x.data<T>()`, POD copies of size/shape/strides). Allows the
work to be queued and run later on a CPU worker thread — by then the original
`mx::array` objects might have moved, but the captured pointers/values stay
valid.

**3. Templated on `T`.** Same pattern as Metal kernel templates: one
implementation, reused across dtypes via explicit specialization.

#### `eval_cpu` — runtime dtype dispatcher

```cpp
void Axpby::eval_cpu(...) {
  if (out.dtype() == mx::float32) return axpby_impl<float>(...);
  else if (out.dtype() == mx::float16) return axpby_impl<mx::float16_t>(...);
  else if (out.dtype() == mx::bfloat16) return axpby_impl<mx::bfloat16_t>(...);
  else if (out.dtype() == mx::complex64) return axpby_impl<mx::complex64_t>(...);
  else throw std::runtime_error("Axpby is only supported for floating point types.");
}
```

The CPU analog of axpby.metal's static instantiation macros: each branch picks
the right templated specialization. This is the canonical pattern for CPU
implementations — same as `mlx/backend/cpu/quantized.cpp`'s `_qmm`/`_qmv`
dispatch.

#### Why we skip CPU for SpQt MVP

1. **Scope.** Single-shape GPU correctness + perf break-even is the bar. Adding CPU adds work without changing what's validated.
2. **Reference comparison is GPU-side.** M2's correctness gate compares against `mx::quantized_matmul` on GPU — CPU isn't in the loop.
3. **MLX-core precedent.** `QuantizedMatmul` is GPU-only for many configurations.

For SpQt:

```cpp
void ZigzagQMVSparse::eval_cpu(...) {
  throw std::runtime_error("ZigzagQMVSparse: no CPU implementation");
}
```

If ever needed, the pattern above is the template — encoder API is reusable
verbatim.

### Part B — Autograd hooks (`:249-298`)

#### `jvp` — forward-mode autodiff

For `z = αx + βy`: `∂z/∂x = α`, `∂z/∂y = β`. So:
- jvp({0}) = α · tangent_x
- jvp({1}) = β · tangent_y
- jvp({0,1}) = α · tangent_x + β · tangent_y = `axpby(tangent_x, tangent_y, α, β)`

axpby's implementation handles all three cases. Notably, **the JVP of axpby
is itself an axpby** — the multi-arg branch dispatches recursively to the
public `axpby(...)` op function.

> ⚠️ Reading axpby's `jvp` body closely, the `if`/`else` branches' conditions
> appear swapped relative to the accompanying comments (the `argnums.size() > 1`
> branch implements the single-arg math, the `else` implements the
> two-arg math). Possibly a bug in the example. Doesn't affect us since we
> throw NYI; if SpQt ever needs an autograd path, derive it fresh rather than
> transliterating from here.

#### `vjp` — reverse-mode autodiff (standard backprop)

For loss `L` with cotangent `∂L/∂z`:
- `∂L/∂x = α · ∂L/∂z`
- `∂L/∂y = β · ∂L/∂z`

Implementation:

```cpp
std::vector<mx::array> Axpby::vjp(...) {
  std::vector<mx::array> vjps;
  for (auto arg : argnums) {
    auto scale = arg == 0 ? alpha_ : beta_;
    auto scale_arr = mx::array(scale, cotangents[0].dtype());
    vjps.push_back(mx::multiply(scale_arr, cotangents[0], stream()));
  }
  return vjps;
}
```

Returns one cotangent per requested `argnum`. The signature also takes the
forward-pass `outputs` for ops where the gradient depends on the output value
(softmax, etc.); axpby doesn't use it.

#### `vmap` — vectorization

axpby itself throws NYI. Optional even for production ops.

#### Why we skip autograd for SpQt MVP

`QuantizedMatmul` in MLX core throws "no gradient wrt the quantized weights"
(`mlx/primitives.cpp:3487`). Quantization isn't differentiable in the usual
sense — round-to-grid is non-differentiable. Standard practice: throw NYI;
training code uses straight-through estimators when needed.

### Recap: which methods we actually implement for SpQt

| Concern | Axpby reference | SpQt M0b/M2 |
|---|---|---|
| `eval_gpu` | Full impl (Order 4) | Full impl (the SpQt kernel) |
| `eval_cpu` | Full impl + dtype dispatch | Throws NYI |
| `jvp` | Full impl, recursive | Throws NYI |
| `vjp` | Full impl, scaled cotangents | Throws NYI |
| `vmap` | Throws NYI | Throws NYI |
| `is_equivalent` | Compares `alpha_`, `beta_` | Compares `group_size_`, `bits_` |
| `name()` | Returns `"Axpby"` | Returns `"ZigzagQMVSparse"` |

We implement **one method substantively** (`eval_gpu`), use **one trivial
method** (`is_equivalent`), and stub out the rest with **explicit NYI throws**.
Three benefits over leaving them unimplemented:

1. **Compiler-checked.** The `override` keyword forces us to provide a body; explicit throws make the intentional non-implementation visible.
2. **Clear runtime error.** Accidental `mx.grad(zigzag_qmv)` gets a helpful message rather than confusing default behavior.
3. **Talk story.** "Forward GPU implemented; the rest explicitly out of scope" is a clean delineation.

### Summary

| Question | Answer |
|---|---|
| Why is the CPU impl symmetric to GPU? | Both go through MLX's `CommandEncoder` API (CPU and GPU variants). Backend-agnostic surface; only the encoder type and dispatch unit (lambda vs Metal kernel) differ. |
| Why does CPU dispatch use a lambda? | The lambda captures data by value so the CPU work can be queued for later execution on a worker thread, after the source `mx::array` objects might have moved. |
| What's a JVP that returns the same op? | axpby's JVP is itself an axpby — beautiful recursive identity for linear ops. Common with most affine operations. |
| Why throw NYI for autograd on quantized ops? | Round-to-grid quantization isn't differentiable; precedent is `QuantizedMatmul::vjp` in MLX core. |
| What does our SpQt extension implement substantively? | Just `eval_gpu` (the kernel) and `is_equivalent` (parameter compare). Everything else throws NYI by design. |

---

## Phase B complete

All seven orders covered. The doc is now a complete walkthrough of the axpby
extension's machinery — useful as a structural reference when writing
`extensions/mlx_spqt/`.

Next: **phase C** (build system) is briefly covered inline in earlier sections
(CMakeLists in Order 5/6, setup.py in Order 6) — could be expanded into its own
doc later if needed.

After that: phase D synthesis happens by *writing* the SpQt extension. The
hands-on construction is the synthesis check.
