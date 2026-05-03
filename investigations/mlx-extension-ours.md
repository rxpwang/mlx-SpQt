# Our MLX extension: `mlx_spqt`

The living record of `extensions/mlx_spqt/` — what's in it, how it got there,
and how each phase of M0b–M2 work extends it. Counterpart to
`mlx-extension-axpby.md` (which is about MLX's upstream example): this doc is
about *our* concrete artifact.

## Current state (TL;DR)

- **Path:** `extensions/mlx_spqt/`
- **Distribution name:** `mlx_spqt`
- **Python import name:** `mlx_spqt`
- **Compiled extension module:** `_spqt_ext`
- **C++ library:** `mlx_spqt` (→ `libmlx_spqt.dylib`)
- **Metal library:** `mlx_spqt` (→ `mlx_spqt.metallib`)
- **C++ namespace:** `spqt_ext`
- **Registered ops:** `axpby` (sanity baseline; per Scope decision #5, will be
  dropped at M4)

End-to-end functional check: `cd extensions/mlx_spqt && python test.py` →
`c_cpu correct: True; c_gpu correct: True`.

## Phase index

| Phase | Status | Content |
|---|---|---|
| 1. Scaffold migration (M0b step 1) | ✅ | Cloned axpby template, renamed to `mlx_spqt`, verified `mlx_spqt.axpby(...)` works |
| 2. MSL feature smoke kernels (M0b steps 2-7) | ☐ | atomic-add, threadgroup memory, uint32 buffer, `qdot` includibility |
| 3. SpQt dense GEMV kernel (M1) | ☐ | `zigzag_quantize` Python helper + dense zigzag-GEMV Metal kernel |
| 4. SpQt sparse-GEMV (M2) | ☐ | idx-driven K-walk added to M1 kernel |
| 5. Cleanup (M4) | ☐ | Drop `axpby/` subdirectory and its bindings |
| 6. Upstream migration (out of scope) | ☐ | Future reference: how the extension would be promoted into MLX core |

---

## Phase 1 — Scaffold migration from axpby (M0b step 1) ✅

Cloned `examples/extensions/` to `extensions/mlx_spqt/`, renamed package-level
identifiers, kept `axpby/` subdirectory intact as sanity baseline (per Scope
decision #5).

### The five name layers

An MLX extension carries five distinct kinds of names — most happen to match
in our case (all `mlx_spqt`), but they live in different systems and serve
different purposes. Confusing them is a common bug source.

| # | Layer | Our identifier | Materializes as |
|---|---|---|---|
| 1 | **Distribution name** (pip metadata) | `mlx_spqt` | `mlx_spqt-X.Y.Z.dist-info/` |
| 2 | **Python import name** (filesystem package) | `mlx_spqt` | `mlx_spqt/__init__.py` directory in site-packages |
| 3 | **Compiled extension module** (nanobind) | `_spqt_ext` | `mlx_spqt/_spqt_ext.cpython-…-darwin.so` |
| 4 | **C++ library target** (CMake) | `mlx_spqt` | `mlx_spqt/libmlx_spqt.dylib` |
| 5 | **Metal library** (runtime lookup) | `mlx_spqt` | `mlx_spqt/mlx_spqt.metallib` |
| (+) C++ namespace (scope-only) | `spqt_ext` | C++ symbols in `libmlx_spqt.dylib` |

### The rename map (axpby → ours)

| Original (in `examples/extensions/`) | Renamed (in `extensions/mlx_spqt/`) | Layer |
|---|---|---|
| `name="mlx_sample_extensions"` | `name="mlx_spqt"` | 1 |
| `packages=["mlx_sample_extensions"]` | `packages=["mlx_spqt"]` | 2 |
| `package_data={"mlx_sample_extensions": …}` | `package_data={"mlx_spqt": …}` | 2 |
| `mlx_sample_extensions/` directory | `mlx_spqt/` directory | 2 |
| `CMakeExtension("mlx_sample_extensions._ext")` | `CMakeExtension("mlx_spqt._spqt_ext")` | 2 + 3 |
| `nanobind_add_module(_ext …)` | `nanobind_add_module(_spqt_ext …)` | 3 |
| `NB_MODULE(_ext, m)` (in bindings.cpp) | `NB_MODULE(_spqt_ext, m)` | 3 |
| `from ._ext import axpby` (in `__init__.py`) | `from ._spqt_ext import axpby` | 3 |
| `project(_ext LANGUAGES CXX)` (in CMakeLists) | `project(_spqt_ext LANGUAGES CXX)` | 3 |
| `add_library(mlx_ext)` | `add_library(mlx_spqt)` | 4 |
| `target_sources(mlx_ext …)` | `target_sources(mlx_spqt …)` | 4 |
| `target_include_directories(mlx_ext …)` | `target_include_directories(mlx_spqt …)` | 4 |
| `target_link_libraries(mlx_ext PUBLIC mlx)` | `target_link_libraries(mlx_spqt PUBLIC mlx)` | 4 |
| `target_link_libraries(_ext PRIVATE mlx_ext)` | `target_link_libraries(_spqt_ext PRIVATE mlx_spqt)` | 3 + 4 |
| `add_dependencies(mlx_ext mlx_ext_metallib)` | `add_dependencies(mlx_spqt mlx_spqt_metallib)` | 4 + 5 |
| `mlx_build_metallib(TARGET mlx_ext_metallib TITLE mlx_ext …)` | `mlx_build_metallib(TARGET mlx_spqt_metallib TITLE mlx_spqt …)` | 5 |
| `d.get_library("mlx_ext", …)` (in `axpby.cpp::eval_gpu`) | `d.get_library("mlx_spqt", …)` | 5 |
| `namespace my_ext { … }` (in `axpby.h`, `axpby.cpp`) | `namespace spqt_ext { … }` | C++ scope only |
| `&my_ext::axpby` (in `bindings.cpp`) | `&spqt_ext::axpby` | C++ scope only |
| `from mlx_sample_extensions import axpby` (in `test.py`) | `from mlx_spqt import axpby` | 2 |

### Files touched

```
extensions/mlx_spqt/
├── setup.py                     ← name, ext_modules, packages, package_data
├── pyproject.toml               ← (no rename — only build-system deps)
├── CMakeLists.txt               ← project, library names, link targets, metallib title
├── bindings.cpp                 ← NB_MODULE name, namespace prefix
├── mlx_spqt/                    ← directory rename (was mlx_sample_extensions/)
│   └── __init__.py              ← import statement
├── axpby/
│   ├── axpby.h                  ← namespace declaration
│   ├── axpby.cpp                ← namespace, d.get_library() string, comment
│   └── axpby.metal              ← (no rename — kernel template + instantiation only)
└── test.py                      ← import statement
```

### Verification commands

After rename + `pip install -e .`, these spot-check every layer:

```bash
# Layer 1 (distribution name)
pip show mlx_spqt | head -1
# expect: Name: mlx_spqt

# Layer 2 (import name resolves; package directory exists)
python -c "import mlx_spqt; print(mlx_spqt.__file__)"
# expect: …/mlx_spqt/__init__.py

# Layer 3 (compiled extension loadable)
python -c "from mlx_spqt._spqt_ext import axpby; print(axpby)"
# expect: <built-in function axpby>

# Layers 4 + 5 (artifacts co-located in the package dir)
python -c "
import mlx_spqt, os
d = os.path.dirname(mlx_spqt.__file__)
print(sorted(f for f in os.listdir(d) if f.endswith(('.so', '.dylib', '.metallib'))))
"
# expect: ['_spqt_ext.cpython-…-darwin.so', 'libmlx_spqt.dylib', 'mlx_spqt.metallib']

# End-to-end functional check (axpby kernel runs on GPU)
cd extensions/mlx_spqt && python test.py
# expect: c_cpu correct: True; c_gpu correct: True
```

### Failure modes for an inconsistent rename

If any layer's identifier doesn't match its dependents, the failure surfaces
at a different stage:

| Mismatch | Symptom | Stage |
|---|---|---|
| `setup.py packages=` doesn't match the on-disk dir | `pip install` fails: "package directory not found" | install |
| `__init__.py from ._spqt_ext` doesn't match `nanobind_add_module(_spqt_ext, ...)` | `import mlx_spqt` fails: `ImportError: No module named 'mlx_spqt._spqt_ext'` | first import |
| `NB_MODULE(_spqt_ext, m)` doesn't match `nanobind_add_module(_spqt_ext, ...)` | dlopen succeeds; Python can't find init function: missing `PyInit__spqt_ext` | first import |
| `target_link_libraries(_spqt_ext PRIVATE mlx_spqt)` doesn't match `add_library(mlx_spqt)` | CMake config error: "target mlx_spqt not found" | configure |
| `mlx_build_metallib(... TITLE mlx_spqt ...)` doesn't match `d.get_library("mlx_spqt", ...)` in axpby.cpp | Build succeeds; first kernel dispatch at runtime fails: "Library not found" | first kernel call |
| `setup.py name=` doesn't match anything else | Cosmetic — pip metadata reads the wrong name; can confuse `pip uninstall` | n/a (just confusing) |
| `test.py` imports the distribution name instead of the import name | `ImportError: No module named 'mlx_spqt_extensions'` | test run |

The last one is what we hit during the migration. Distribution name (used by
pip) and import name (used by Python) are independent in setuptools and
**conventional, but not required, to match**. Our fix was to make them match
(`name="mlx_spqt"` in setup.py).

### One stale comment surviving the rename

`extensions/mlx_spqt/axpby/axpby.cpp:21` still references the old metallib
filename:

```cpp
// The Metal library ("mlx_ext.mtllib"), should be in the same directory.   ← stale
```

Should be:

```cpp
// The Metal library ("mlx_spqt.metallib") should be in the same directory.
```

(The original also had a typo — `.mtllib` for `.metallib` — worth fixing on
the same pass.) Comment-only; no behavioral impact.

### Lessons applicable to any extension rename

1. **Five name layers, not one.** Enumerate the layers and verify each
   independently. Don't bulk-sed; rename each layer deliberately and confirm
   with `pip show` / `python -c "import …"` before moving on.

2. **Distribution name vs. import name.** setuptools allows them to differ
   (e.g. `pip install scikit-learn` → `import sklearn`). Match them unless you
   have a reason not to — the asymmetry is a constant trap.

3. **`d.get_library(name, dir)` is the runtime gate.** The metallib's `TITLE`
   in CMake and the string passed to `d.get_library(...)` in C++ must match
   exactly. Easy to miss because it's buried in `eval_gpu`'s body, not in a
   build-system file.

4. **The C++ namespace rename is optional but worth doing.** Keeps the source
   from looking like forgotten leftover. Three files: `axpby.h`, `axpby.cpp`,
   `bindings.cpp`.

5. **Stale comments survive bulk renames.** Comments referring to old
   identifiers won't be flagged by any compiler error. Grep for old names in
   comments after every bulk rename.

---

## Phase 2 — MSL feature smoke kernels (M0b steps 2-7) ✅

Four smoke kernels added under `extensions/mlx_spqt/`, each in its own
subdirectory mirroring axpby's pattern. All four pass their correctness
gates. Per the Phase index above, M0b is now complete.

### What we built

| # | Subdir | Tests | M2 capability gated | Result |
|---|---|---|---|---|
| 1 | `smoke_uint32/` | `test_smoke_uint32.py` | `mx.array(dtype=mx.uint32)` round-trip; nibble extraction | nibbles 0..15 from `[0x76543210, 0xFEDCBA98]` round-trip correct |
| 2 | `smoke_qdot/` | `test_smoke_qdot.py` | `qdot` / `load_vector` from `quantized.h` callable (Working Principle #4) | `out[0] = 120.0` (exact dot-product result) |
| 3 | `smoke_atomic/` | `test_smoke_atomic.py` | cross-TG `atomic_fetch_add_explicit` on `device atomic_float*` | 256 TGs × 32 threads → `out[0] = 8192.0` exact (no contention loss) |
| 4 | `smoke_threadgroup/` | `test_smoke_threadgroup.py` | within-TG cross-simdgroup reduce via `threadgroup` memory + `threadgroup_barrier` | 4 SGs × 32 lanes → `out[0] = 128.0` |

Each smoke kernel has its own `Primitive` subclass, `eval_gpu`, and Metal
source. axpby continues to work in parallel as the sanity baseline (per
Scope decision #5).

### Working Principle #4 verified

The most important outcome: **`#include "mlx/backend/metal/kernels/quantized.h"`
works from extension `.metal` source** when paired with the right transitive
includes:

```metal
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"      // ← required for steel:: types
#include "mlx/backend/metal/kernels/quantized_utils.h"      // ← required for elem_to_loc_broadcast etc.
#include "mlx/backend/metal/kernels/quantized.h"
```

This is the same include pattern MLX itself uses in `quantized.metal:1-7`. With
these in place, `qdot` / `load_vector` / `get_pack_factor` are all callable
from our extension's source — M2 can use them directly without vendoring.

### Failure modes encountered (lessons for M2)

The four smoke kernels surfaced a variety of failure modes worth cataloging
before M2's larger kernel work. Each was small and locally diagnosable —
exactly the value of doing them as separate, focused exercises.

**Build-system / CMake-level**

| Failure | Diagnosis | Fix |
|---|---|---|
| `No rule to make target '.../smoke_xxx.metal$'` (stray `$`) | Editor inserted a stray character into CMakeLists path | Remove the `$`; CMake errors clearly identify the bad path |
| `No rule to make target '.../smoke_uint32/smoke_qdot.metal'` | Wrong directory in path (copy-paste from sibling kernel) | Fix the subdir name in `mlx_build_metallib SOURCES` |
| Stale metallib after `.metal` edits — kernel runs old code | CMake editable install doesn't always detect `.metal` changes | `rm -rf build && pip install -e . --force-reinstall --no-deps` for clean rebuild |
| `'mlx/backend/metal/kernels/quantized.h' file not found` (initially expected; actually OK) | Path *is* on `MLX_INCLUDE_DIRS` but transitive deps aren't | Add `steel/gemm/gemm.h` and `quantized_utils.h` *before* `quantized.h` |
| Build succeeds but runtime "Library not found: mlx_xxx.metallib" | `mlx_build_metallib(... TITLE ...)` doesn't match `d.get_library("...", ...)` string | Make them match; both layers' Metal-library-name must align |

**Compile / language-level**

| Failure | Diagnosis | Fix |
|---|---|---|
| `no member named 'set_constant_array' in 'CommandEncoder'` | Method doesn't exist; `set_bytes` is the right one for inline scalars | Use `set_bytes(value, idx)` for scalars; `set_input_array`/`set_output_array` for `mx::array`s |
| `use of undeclared identifier 'atomic_fetch_add'` | MSL only has the `_explicit` variant of atomics | Use `atomic_fetch_add_explicit(out, val, memory_order_relaxed)` |
| `no member named 'smoke_xxx' in namespace 'spqt_ext'` | Forgot to `#include "smoke_xxx/smoke_xxx.h"` in `bindings.cpp` | Add the include |
| `duplicate symbol: spqt_ext::current_binary_dir` | Two `.cpp` files defining the same external-linkage helper | Wrap each copy in anonymous namespace for file-local linkage |
| `mx::array(int, ...)`: shape constructor doesn't match `int` | `mx::Shape` is a vector type; needs `{N}` not `N` | Use brace-init `{N}` |
| `use of undeclared identifier 'w_packed'` in `eval_gpu` | Stale copy-paste from a different smoke kernel; references nonexistent variables | Adapt op function and eval_gpu carefully per kernel; don't bulk-copy |
| Wrong include `axpby/axpby.h` instead of own header | Copy-paste leftover | Use the correct `smoke_xxx/smoke_xxx.h` include |

**Semantic / runtime**

| Failure | Diagnosis | Fix |
|---|---|---|
| Output shape `(N,)` when only `out[0]` is written | Op function returned `{N}` instead of `{1}` | Either use `{1}` or use `out[0].item()` in tests |
| `out.item()` raises "Only length-1 arrays can be converted" | Output shape isn't `(1,)` | Either fix the op function's shape, or use `out[0].item()` |
| `dispatch_threadgroups` vs `dispatch_threads` confusion | `dispatch_threads(grid, tgp)` interprets `grid` as TOTAL threads; `dispatch_threadgroups(grid, tgp)` interprets `grid` as TG count | Pick the variant that matches your mental framing; we use `dispatch_threads` for "N output elements ⇒ N threads" |
| Distribution name vs import name confusion in setup.py | `name=` is pip metadata; `packages=` is Python import name; can differ | Match them — `name="mlx_spqt"` and `packages=["mlx_spqt"]` |

### General lessons applicable to M2

1. **`pip install -e . --force-reinstall --no-deps` after `.metal` edits.** CMake's
   editable-install dependency tracking on `.metal` files is unreliable. Force-rebuild
   when in doubt; it's cheap (a few seconds) compared to debugging a stale-kernel mystery.

2. **Buffer-slot consistency is the silent-corruption risk.** Mismatched `[[buffer(N)]]`
   in `.metal` vs slot index in `set_*_array(arr, N)` produces no error; just garbage
   output. Verify by treating the kernel signature and the eval_gpu binding as a single
   contract and reviewing them together.

3. **MSL atomics scale.** `atomic_fetch_add_explicit` is exact and lossless under
   meaningful contention (256 TGs × 32 threads, all targeting the same cell). M2's
   K-tiled architecture can rely on cross-TG atomic-add for row-partial reduction
   without scaling concerns at our shape.

4. **MLX's `qdot` / `load_vector` pair is callable from extensions.** Their math
   semantics are intricate (especially the bit-position trick + pre-scaling for
   `bits=4`), but they Just Work when called as a pair per MLX's canonical pattern.
   Don't try to call `qdot` without the matching `load_vector` setup.

5. **Don't bulk-copy `eval_gpu` between smoke kernels.** Each kernel has slightly
   different inputs/outputs/dispatch geometry. Stale variable references and slot
   indices are the most common bug class — visible at compile time, but tedious to
   debug if you've copied 50 lines.

6. **Failure modes are *local*.** Picking small, focused smoke kernels paid off — every
   compile/runtime error was one or two lines wrong, with a clear diagnostic. Compare
   to a hypothetical combined kernel where a single failure could be in any of four
   features. The MILESTONES decision to split into four was the right one.

---

## Phase 3 — SpQt dense GEMV kernel (M1) ☐ (in progress)

The kernel-design milestone. Adds `zigzag_quantize` (Python helper) plus the
dense zigzag-GEMV Metal kernel. M2 then increments this with idx-driven
sparsity.

### M1 deliverables (per `MILESTONES.md`)

1. `zigzag_quantize(w_fp, group_size, bits) → (w_zz, scales_zz, biases_zz)` —
   pure Python/MLX helper. Takes the *original* full-precision weight matrix
   and returns zigzag-quantized output (rearrange-then-quantize, matching the
   llama.cpp-SpQt reference's pattern).
2. **Dense** zigzag-GEMV Metal kernel — full multi-TG K-tiled architecture
   with cross-TG atomic-reduce. Walks all K-positions contiguously (no
   sparsity yet; that lands in M2).
3. Correctness gate: `(y - y_ref).abs().max() < 1e-3` vs.
   `mx.quantized_matmul` on equivalent (non-zigzag) packed weights, single
   shape, dense activations.

### Why M1 needs design before code

Three layout decisions interlock — getting any of them wrong means redoing
work:

- **`zigzag_quantize` and the kernel must agree on the exact memory layout**
  of `w_zz` / `scales_zz` / `biases_zz`. Subtle disagreement = silent buffer-
  binding corruption.
- **Whether `qdot` / `load_vector` reuse fits** depends on the layout's
  consumption pattern. Working Principle #4 wants reuse, but the SpQt
  reference's "one activation × many row-partials" inversion may not match
  qmv_fast's "many activations × few row-partials" pattern.
- **Threadgroup geometry** (rows-per-TG = `group_size`, simdgroups per TG,
  K-stride per simdgroup) determines memory access patterns and
  atomic-contention cost. Has to be picked once and consistent across kernel
  + dispatch.

We design before code so these decisions are explicit and reviewed.

### Step plan

| Step | Output | Owner | Status |
|---|---|---|---|
| 3.1 | `investigations/mlx-spqt-reference.md` — what zigzag concretely means in llama.cpp-SpQt's reference (fp-space rearrangement + dense kernel walk) | Claude (investigation) | ✅ |
| 3.2 | `investigations/mlx-qmv-fast.md` — line-by-line deep-dive of `qmv_fast_impl` to identify what we mirror vs. flip | Claude (investigation) | ☐ |
| 3.3 | `investigations/M1-zigzag-layout.md` — design spec combining 3.1 and 3.2: layout choice, kernel architecture, qdot-reuse decision, threadgroup geometry | Claude (investigation) | ☐ |
| 3.4 | `extensions/mlx_spqt/mlx_spqt/zigzag_quantize.py` (or in `__init__.py`) — Python/MLX implementation; tested in isolation via dequantize round-trip | User (with line-level instruction) | ☐ |
| 3.5 | `extensions/mlx_spqt/zigzag_qmv_dense/` extension — `.h` + `.cpp` + `.metal` + binding + CMakeLists wiring; mirrors the M0b add-a-kernel workflow | User | ☐ |
| 3.6 | `extensions/mlx_spqt/test_zigzag_qmv_dense.py` — correctness check against `mx.quantized_matmul` on equivalent (non-zigzag) weights | User | ☐ |

### Why this order

- **3.1 first** anchors what zigzag concretely is (target pattern). Sets the
  vocabulary: super-block-per-input-column, fp-rearrange-then-quantize, etc.
- **3.2 next** anchors the structural template (`qmv_fast_impl`'s patterns).
  Now we can talk about the M1 kernel as "qmv_fast with these specific
  changes" rather than abstractly.
- **3.3 commits to a design.** Concrete layout, geometry, and qdot decision.
  After this, 3.4-3.6 are bounded effort with no design risk.
- **3.4 before 3.5** — Python is faster to iterate on, easier to verify in
  isolation (dequantize round-trip, compare to original up to quant error).
  Locks in the layout `zigzag_quantize` produces *before* the kernel has to
  consume it.
- **3.5 before 3.6** — kernel scaffold + body + eval_gpu before the
  correctness test. Mirrors M0b workflow.

### Risks specific to M1

| Risk | Mitigation |
|---|---|
| `qdot`'s consumption pattern doesn't fit the zigzag layout | Resolved in 3.3 with the explicit qdot-reuse decision. Fallback: write a custom inner loop without qdot (lose Working Principle #4 partial credit; still correct). |
| Threadgroup geometry choice is suboptimal for performance | M1 only requires correctness, not performance. Geometry is a tuning knob revisited in M3. |
| `zigzag_quantize`'s output layout subtly disagrees with the kernel's expectation | Test 3.4 in isolation (round-trip via `mx.dequantize`) to lock the layout before kernel work begins. |
| Cross-TG atomic-add accumulator needs explicit zero-fill | M0b smoke #3 confirmed `atomic_fetch_add_explicit` works at scale. Zero-fill is an `eval_gpu`-side detail (either explicit kernel or trust `mx::allocator::malloc` zeroing — to be confirmed in 3.5). |

### Success criteria for M1 (= done criterion)

```python
# w is the original fp16 weight matrix (single shape: M=4096, K=4096)
w_zz, scales_zz, biases_zz = mlx_spqt.zigzag_quantize(w, group_size=64, bits=4)
y_zz = mlx_spqt.dense_zigzag_qmv(x, w_zz, scales_zz, biases_zz)

# reference: standard (non-zigzag) quantization of the same weights
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
y_ref = mx.quantized_matmul(x, w_q, scales, biases, transpose=True,
                             group_size=64, bits=4)

assert (y_zz - y_ref).abs().max().item() < 1e-3
```

Both sides quantize the same `w`; both compute `x @ dequant(w).T` modulo
quantization error. The only difference between sides is the layout and the
kernel — if they match within MLX's standard tolerance, M1's contract holds.

### What M1 does NOT need to deliver

- **No sparsity.** Idx-driven K-walk lands in M2.
- **No performance tuning.** Threadgroup geometry can be picked for ease of
  implementation, not perf. M3 revisits.
- **No multi-shape support.** Single fixed shape (M=4096, K=4096, fp16,
  gs=64, b=4) per Scope decision #3.
- **No CPU implementation.** GPU-only forward, throw NYI on CPU.
- **No autograd.** Throws NYI on jvp/vjp/vmap (matches `QuantizedMatmul` in
  MLX core).

---

## Phase 4 — SpQt sparse GEMV (M2) ☐

*To be filled in during M2.* Idx-driven K-walk added to M1's kernel.

---

## Phase 5 — Cleanup (M4) ☐

*To be filled in during M4.* Drop the `axpby/` subdirectory and its
`bindings.cpp` registration. Final extension contains only SpQt code.

---

## Phase 6 — Upstream migration (out of scope; future reference)

If `mlx_spqt` ever gets promoted into MLX core (e.g. as
`mx.quantized_matmul(..., mode="affine_zigzag")`), the migration is
well-scoped. Most of our structural choices (Working Principle #4 mirroring
`qmv_fast_impl`, the `Primitive` shape matching `QuantizedMatmul`, the
`eval_gpu` pattern) make this a lift-and-shift rather than a rewrite.

**Effort estimate:** ~11-19 hours of focused work
(*revised from earlier ~8-15h estimate after Phase 3.1 SpQt-reference investigation*).

### What's been clarified through M1 investigation

Two findings that meaningfully shifted the picture:

1. **Zigzag is fp-space rearrangement + standard quantize, not a packed-bytes operation.**
   Per `mlx-spqt-reference.md`, `rearrange_tensor_zigzag` is a 12-line fp
   reshuffle that runs *before* the unchanged Q4_K quantizer. For upstream:
   `mx.quantize(mode="affine_zigzag")` becomes "reshape input → call
   existing `affine_quantize`" — much simpler than I'd estimated. **~1-2h
   instead of ~2-4h.**

2. **The dense kernel is a structural inversion of qmv_fast, not a
   parameter tweak.** It's "one activation × many row-partials per thread"
   (vs. qmv_fast's "many activations × few row-partials"). It's a *new*
   kernel template alongside `qmv_fast_impl`, but it slots cleanly into the
   existing template-cascade and macro infrastructure.

### Component-by-component migration map (updated)

| Extension piece | Core counterpart | Difficulty | Hours |
|---|---|---|---|
| `zigzag_quantize` (Python) | Port to C++ behind `mx.quantize(mode="affine_zigzag")`. Reshape + existing `affine_quantize` machinery — *not* a separate algorithm | **Low** (was Medium) | 1-2 |
| Mode-dispatch wiring | Add `QuantizationMode::AffineZigzag` enum; touch `string_to_quantization_mode`, `quantization_mode_to_string`, `quantized_matmul` op function, `qmv` dispatcher | Medium | 2-3 |
| Metal kernel template + macro instantiation | Insert new template alongside `qmv_fast_impl` in `mlx/backend/metal/kernels/quantized.h`; add to `quantized.metal` cascade | Low — mirrors qmv_fast template structure | 2-3 |
| `eval_gpu` wrapper / dispatcher | New function in `mlx/backend/metal/quantized.cpp` for the zigzag kernel | Low/Medium — cross-TG accumulator complicates dispatcher (see new risk below) | 2-3 |
| **Cross-TG atomic accumulator infrastructure** (new) | Pre-zeroed scratch buffers + "last-TG-arrives" pattern. MLX's existing dispatchers don't have this convention; needs new infrastructure (or alternative reduction strategy). | **New / Medium** | 2-4 |
| **`atomic_float` vs. int-scaling fallback** (new, conditional) | We use `atomic_float` directly (M0b smoke #3 verified). Reference uses int-scaling for portability across older Apple GPUs. Maintainers may want both paths. | **New / conditional** | +1-2 |
| Tests | Slot into `python/tests/test_quantized.py` following the `test_qmv` pattern | Trivial | 0.5-1 |
| `bindings.cpp`, `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `current_binary_dir()` | Deleted; MLX's existing machinery covers them | Trivial | 0 |
| **Total** | | | **11-19 hours** |

### Newly identified risks (not visible in earlier Phase 6)

#### 1. Cross-TG atomic accumulator infrastructure

The SpQt reference allocates `atomic_array` and `atomic_counter` as device-side
scratch buffers, expects them **pre-zeroed before each call**, and uses a
"last-TG-arrives" pattern to write the final result. MLX core's existing
quantized dispatchers (`qmv`, `qmm`, etc.) **don't have this pattern** — they
go single-pass with no cross-TG reduction.

For upstream, two paths:

- **(a) Establish a "scratch-buffer-with-zero-fill" convention in
  `quantized.cpp`.** The dispatcher allocates the atomic buffers, zero-fills
  them (separate pre-kernel or trust allocator-zeroing), then dispatches the
  main kernel. *New infrastructure for MLX's quantized path.*
- **(b) Use a different reduction strategy** — e.g., 2-pass: per-TG outputs
  to an intermediate buffer, then a separate reduction kernel. Cleaner
  architecturally but adds a kernel and an allocation.

Either path adds 2-4h of infrastructure work beyond a "drop-in kernel" port.
Worth surfacing during code review with MLX maintainers — they may have an
opinion on which pattern fits their architectural conventions.

#### 2. `atomic_float` vs. int-scaling tradeoff

We use `atomic_fetch_add_explicit` on `device atomic_float*` directly
(verified to scale via M0b smoke #3 — 256 TGs × 32 threads, lossless on
M-series). The SpQt reference uses an int-scaling workaround
(`SCALE_FACTOR = 1e5`, lossy ~5 decimal digits) because **`atomic_float` is
software-emulated on some older Apple GPUs**.

For upstream:
- MLX core supports a wide hardware range. Maintainers may want **both
  paths** for portability: `atomic_float` on hardware that supports it
  natively, int-scaling fallback elsewhere.
- That's a runtime-or-build-time toggle plus the int-scaling workaround
  code. Conditional cost: +1-2h. Skippable if maintainers accept atomic_float
  as the only path.

### M1 design decisions that affect upstream cost

These choices, made now in M1's design doc (Step 3.3), have downstream
implications for upstream-migration cost:

1. **Use `atomic_float` directly, or hedge with int-scaling fallback now?**
   - Going atomic_float-only: simpler M1, faster perf in our test
     environment, may need rework for upstream merge.
   - Hedging with int-scaling toggle now: more code in M1, but
     upstream-ready out of the box.
   - **M1 recommendation**: **atomic_float-only for M1/M2/M3** (simpler,
     faster, valid on our test hardware). If we ever upstream, add the
     int-scaling fallback at that point.

2. **How to handle cross-TG accumulator zero-fill?**
   - Trust `mx::allocator::malloc` to return zeroed memory: cleaner code,
     depends on MLX implementation details we observed empirically.
   - Explicit zero-fill in `eval_gpu`: more robust, ports cleanly to upstream
     where MLX's behavior may differ.
   - **M1 recommendation**: **explicit zero-fill** (small extra kernel or
     `MTLBlitCommandEncoder memset`). Saves rework for upstream.

### Unchanged from earlier Phase 6

#### Two API design questions still deferred to upstream merge time

1. **How to expose `idx`** in the public API (M2-relevant):
   - Option α: optional `idx` kwarg on `mx.quantized_matmul` (idiomatic;
     subtle API surface change).
   - Option β: separate `mx.zigzag_quantized_matmul(...)` op (uglier but no
     public-API risk).

2. **Where dense-zigzag vs. sparse-zigzag dispatch lives:** single
   `mode="affine_zigzag"` with optional `idx`, or two modes
   (`"affine_zigzag"` / `"affine_zigzag_sparse"`).

Both questions are deferred. They wouldn't change the kernel work, only the
public-facing wrapper.

#### Structural choices that keep upstream-cost low

- **Working Principle #4 (mirror qmv_fast)** — our kernel slots into the
  existing template/macro/dispatch tree without restructuring.
- **MLX-affine over Q4_K** — flat per-group scales (no super-block
  hierarchy) means no Q4_K-specific decoding logic to port. We avoid most of
  llama.cpp-SpQt's complexity.
- **The fp-rearrange-then-quantize separation** — keeps `zigzag_quantize`
  independent of the kernel; can be implemented and tested in isolation,
  both as extension and as core.

### Overall verdict

**Migration risk is moderate and well-bounded.** Estimate ≈ 11-19 hours,
similar magnitude to Phase 6's earlier 8-15h estimate but with a modestly
shifted distribution (easier on Python-port side, slightly harder on
kernel-infrastructure side). The structural choices we've already locked in
(Working Principle #4, MLX-affine target, extension-first development) keep
cost low. The newly identified risks (cross-TG accumulator infra,
atomic_float portability) are real but each is bounded to a few hours of
well-defined work.

**No blockers for upstream merge become visible from this analysis.** The
path stays clear.

### Talk-friendly framing

> Built as a registered MLX extension for the MVP. Upstream migration is
> well-scoped (~11-19 hours, mostly kernel + mode-dispatch wiring); the
> kernel itself slots into MLX's existing template structure with two new
> infrastructure pieces (cross-TG accumulator pattern; atomic_float
> portability fallback). The extension/core boundary was chosen deliberately
> so the MVP shipped quickly and the upstream path isn't a rewrite.

Maps cleanly onto the rubric's "minimal effort to MVP, clean path to
production" framing.
