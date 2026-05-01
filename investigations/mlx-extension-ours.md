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

## Phase 2 — MSL feature smoke kernels (M0b steps 2-7) ☐

*To be filled in as smoke kernels are added.* Per `MILESTONES.md` M0b detail,
four smoke kernels gate the M2 design:

- [ ] Atomic-add into `device atomic_float*` output
- [ ] `threadgroup` memory + cross-simdgroup reduce
- [ ] `uint32`-packed input round-trip
- [ ] `#include "mlx/backend/metal/kernels/quantized.h"` + `qdot<…>` callable

Each adds a new kernel under `extensions/mlx_spqt/`, registered through the
shared `bindings.cpp` and built into the same metallib. axpby continues to
work in parallel.

---

## Phase 3 — SpQt dense GEMV kernel (M1) ☐

*To be filled in during M1.* Adds `zigzag_quantize` (Python helper) plus the
dense zigzag-GEMV Metal kernel. The kernel-design milestone.

---

## Phase 4 — SpQt sparse GEMV (M2) ☐

*To be filled in during M2.* Idx-driven K-walk added to M1's kernel.

---

## Phase 5 — Cleanup (M4) ☐

*To be filled in during M4.* Drop the `axpby/` subdirectory and its
`bindings.cpp` registration. Final extension contains only SpQt code.
