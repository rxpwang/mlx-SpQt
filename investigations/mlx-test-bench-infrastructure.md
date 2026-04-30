# MLX test & benchmark infrastructure

A pointer-heavy survey of how MLX itself tests its kernels and benchmarks them. Used as a starting reference for our M0b feasibility checks, M2 correctness tests, and M3 perf benches.

All file/line references are against `rxpwang/spqt` at `e8ebdebe` (vanilla upstream MLX, post-0.31.2).

## TL;DR

- **Quantized correctness convention is `max-abs-diff < 1e-3`** (`test_quantized.py:355`) — *not* NMSE. Our M2 gate should align.
- **Bench convention is `time_fn` from `benchmarks/python/time_utils.py`:** 5 warmup, 100 timed iters, mean ms per `fn` call. Uses `mx.eval` (no explicit `mx.synchronize`). **What `fn` returns determines per-call vs. throughput:** one op → per-call latency; list of N un-eval'd ops → throughput (per-op = reported msec ÷ N). Pick consciously.
- **`mx.fast.metal_kernel` is also well-tested** (`test_fast.py`, 6 tests covering multi-I/O, mixed dtypes, templates, strides). Useful as reference for what MSL features are usable, but **we're going Route B** (extension-with-Primitive); the JIT route is *not* our M0b path.

## 1. Test infrastructure

### File layout

| Path | What |
|---|---|
| `python/tests/` | Python tests, one file per topic; `unittest.TestCase` style |
| `python/tests/mlx_tests.py` | shared `MLXTestCase` base |
| `python/tests/__main__.py` | runner |
| `tests/` | C++ tests for core (lower level; not directly relevant for us) |

Run with `python -m unittest discover python/tests` (or via `pytest`).

### Quantized-op correctness pattern

`python/tests/test_quantized.py` is the closest template for our M2 correctness test. Key methods:

| Test | Lines | What it does |
|---|---|---|
| `test_qmv` | `:330-355` | quantized matrix–vector for affine mode; sweeps `(group_size, bits, M, N, B)` |
| `test_qmm` | `:208-244` | quantized matrix–matrix |
| `test_qmm_vjp` | `:245-274` | vjp |
| `test_qmv_small_non_multiples` | `:697-` | small/awkward shapes |

The standard correctness pattern (lifted from `test_qmv:330-355`):

```python
def test_qmv(self):
    key = mx.random.key(0)
    k1, k2 = mx.random.split(key)
    tests = product(
        [128, 64, 32],   # group_size
        [2, 3, 4, 5, 6, 8],   # bits
        [256, 512, 67],   # M
        [64, 256],   # N
        [0, 1, 3, 8],   # B
    )
    for group_size, bits, M, N, B in tests:
        if group_size > N:
            continue
        with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
            x_shape = (3, 1, N) if B == 0 else (B, 1, N)
            w_shape = (M, N) if B == 0 else (B, M, N)
            x = mx.random.normal(shape=x_shape, key=k1)
            w = mx.random.normal(shape=w_shape, key=k2)
            w_q, scales, biases = mx.quantize(w, group_size, bits)
            w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, True, group_size, bits)
            y_hat = x @ mx.swapaxes(w_hat, -1, -2)
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)   # ★ correctness gate
```

**Key points:**

- **Correctness gate:** `(y - y_ref).abs().max() < 1e-3`. Max absolute element error against a dequantized reference. Not NMSE.
- **Reference is dequant-then-FP-matmul.** `w_hat = dequantize(w_q, ...)` then `x @ w_hat.T`. This is what we'd compute in M1 (the dense reference qmv against repacked weights — verifies the layout — and what M2 compares against with sparse columns of `x` zeroed).
- **`subTest` for parameter sweeps.** Lets one method cover the cartesian product without the loop hiding which case failed.
- **MLX random with key splitting** for reproducibility.

For our M2: only one `subTest` (single shape), but mirror the structure.

### Custom Metal kernel pattern

`python/tests/test_fast.py` has the canonical `mx.fast.metal_kernel` examples. Three tests fully cover the API surface:

| Test | Lines | What it verifies |
|---|---|---|
| `test_custom_kernel_basic` | `:740-770` | minimum end-to-end: 1 input, 1 output, fp32, 1D dispatch, `mx.allclose` check |
| `test_custom_kernel_args` | `:773-829` | multi-input + multi-output + mixed dtypes (fp32, bf16, int) + scalar Python args + `template` for compile-time consts |
| `test_custom_kernel_strides` | `:832-` | `inp_shape`, `inp_strides`, `inp_ndim` plumbed automatically; `elem_to_loc` helper available |

The `mx.fast.metal_kernel` API surface (from these tests):

```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",                  # str
    input_names=["x", "scales", ...],  # list[str], match buffers in MSL
    output_names=["out"],              # list[str]
    source=msl_source,                 # str — MSL function body (no [[kernel]] wrapper needed)
)

(out,) = kernel(
    inputs=[x, scales, ...],           # list[mx.array | scalar]
    template=[                         # list[(name, value)] — substituted as compile-time consts/types
        ("T", mx.float16),
        ("GROUP_SIZE", 64),
        ("BITS", 4),
    ],
    grid=(N, 1, 1),                    # tuple[int, int, int]
    threadgroup=(SIMD_SIZE, NSG, 1),   # tuple[int, int, int]
    output_shapes=[(M,)],              # list[tuple]
    output_dtypes=[mx.float16],        # list[dtype]
    stream=mx.gpu,                     # optional, default-stream by default
)
```

**Things verified by these tests:**

- Multi-dtype I/O including `bf16`, `int32`, scalar Python `float`/`int`/`bool`.
- `template=[...]` for compile-time specialization (analog of MLX's `<T, gs, bits>` template params).
- `thread_position_in_grid` and other Metal thread attrs available directly.
- Stride/shape buffers (`inp_shape`, `inp_strides`, `inp_ndim`) injected when needed.
- `elem_to_loc(elem, inp_shape, inp_strides, inp_ndim)` helper available in MSL source.
- `stream=mx.gpu` is honored — kernel runs on the chosen MLX stream.

**Things *not* verified by these tests** (the gaps M0b targets):

- Atomic ops on `device float*` outputs (for cross-TG row-partial reduce in M2).
- `threadgroup` memory qualifier for cross-simdgroup reduction within a TG.
- `uint32`-packed weight buffers as inputs (likely fine — it's just `mx.array(dtype=mx.uint32)` — but worth a smoke test).

Everything else on the M0b checklist is implicitly proven; we don't need to re-verify it.

## 2. Benchmark infrastructure

### File layout

| Path | What |
|---|---|
| `benchmarks/python/` | Python benches, one per op family |
| `benchmarks/python/time_utils.py` | shared timing utilities |
| `benchmarks/python/blas/` | GEMM / GEMV (against torch.mps) |
| `benchmarks/python/comparative/` | side-by-side MLX vs torch |
| `benchmarks/cpp/` | C++ benches (lower level) |

### The standard timing utility

`benchmarks/python/time_utils.py:8-26`:

```python
def time_fn(fn, *args, **kwargs):
    msg = kwargs.pop("msg", None)
    if msg:
        print(f"Timing {msg} ...", end=" ")
    else:
        print(f"Timing {fn.__name__} ...", end=" ")

    # warmup
    for _ in range(5):
        mx.eval(fn(*args, **kwargs))

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    msec = 1e3 * (toc - tic) / num_iters
    print(f"{msec:.5f} msec")
```

Also `measure_runtime(fn, **kwargs)` (`:29-38`) returns the number instead of printing.

**Conventions baked in:**

- **5 warmup iterations**, then **100 timed iterations**.
- **Reports mean ms** per `fn` call (not median, no p-stats by default).
- **Wait via `mx.eval` only** — no `mx.synchronize()`. (Per the execution-model dive, `mx.eval` already blocks on the synchronizer event; `mx.synchronize` is redundant for this usage.)
- **What `fn` returns decides per-call vs. throughput.** `time_fn`'s outer `mx.eval(fn(...))` materializes whatever `fn` produced:
  - `fn` returns **one `mx.array`** (one op) → **per-call latency**, one kernel per command buffer per iteration.
  - `fn` returns **a list of N un-eval'd `mx.array`s** → **batched throughput**, N ops share one (or a few) command buffers; per-op time = reported msec ÷ N.
  - `fn` itself calls `mx.eval(ys)` internally and returns ys → equivalent to the batched case (the outer eval is a no-op).
  
  `bench_gemv.py:48-54` uses the "internal eval + return list" variant. `gather_qmm_bench.py` uses the per-call variant. Both valid; the choice is what you're measuring, not which is "right".

### Quantized-op bench example

`benchmarks/python/gather_qmm_bench.py` is the closest analog to what we need (quantized matmul, time_fn-based). Pattern:

```python
def time_gather_qmm():
    x = mx.random.normal(...)
    w1 = mx.quantize(mx.random.normal(...))
    indices = ...
    mx.eval(x, w1, indices)   # materialize inputs out-of-band — not part of timing

    def gather_mm(x, w1, indices, sort):
        x = mx.gather_qmm(x, *w1, transpose=True, rhs_indices=idx, sorted_indices=sort)
        ...
        return x

    for sort in [True, False]:
        time_fn(gather_mm, x, w1, indices, sort, msg=f"sort={sort}")
```

Two takeaways:

1. **Inputs materialized before timing.** `mx.eval(x, w1, ...)` runs before `time_fn` so the timed region only contains the op under test.
2. **`time_fn` called with the inner function and its args.** Multiple variant configs are timed by repeated `time_fn` calls with different args.

### GEMV bench (closest perf shape)

`benchmarks/python/blas/bench_gemv.py` benches dense GEMV against `torch.mps`. Note its `bench()` function (`:36-45`):

```python
def bench(f, m, v):
    for i in range(N_warmup):
        f(m, v)
    torch.mps.synchronize()
    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(m, v)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9
```

Different style: `torch.mps.synchronize()` is used because torch is in the loop too. For pure-MLX benches, `time_fn` is the idiomatic pattern.

## 3. The API surface for what M0b/M2/M3 will actually use

We go with Route B (extension-with-Primitive); the kernel is registered as
`mlx_spqt.zigzag_qmv` via the axpby-template extension. From Python it looks like
any other MLX op — returns an `mx.array`, evaluates lazily, runs on the default
GPU stream.

### Invocation site (M2/M3)

```python
import mlx.core as mx
import mlx_spqt   # our extension; pip install -e extensions/mlx_spqt/

# w_zz, scales_zz, biases_zz are produced by mlx_spqt.zigzag_repack(...) (M1)
# idx is a host-built mx.array of int32 non-sparse K-indices
y = mlx_spqt.zigzag_qmv(x, w_zz, scales_zz, biases_zz, idx)
```

### M2 correctness test

Mirror `test_quantized.py::test_qmv:330-355`:

```python
import unittest
import numpy as np
import mlx.core as mx
import mlx_spqt


class TestZigzagQMV(unittest.TestCase):
    def test_zigzag_qmv_sparse(self):
        # build inputs (single shape per Scope decision #3)
        # quantize w; repack into zigzag layout via mlx_spqt.zigzag_repack
        # build sparse mask + idx host-side (np.flatnonzero)
        # y_zz = mlx_spqt.zigzag_qmv(x, w_zz, scales_zz, biases_zz, idx)
        # y_ref = mx.quantized_matmul(x_sparse, w_q, scales, biases, transpose=True, ...)
        #         where x_sparse is x with sparse cols zeroed
        self.assertLess((y_zz - y_ref).abs().max().item(), 1e-3)
```

### M3 bench: closure-factory pattern for both modes

`time_fn` doesn't itself batch — `fn` does. The cleanest way to bench both
per-call latency and throughput against the same op is to wrap a single-op
closure in two different `fn`s:

```python
from time_utils import time_fn   # vendored from MLX's benchmarks/python/

# ----- single-op closure factories (one per variant under test) -----
def make_stock(x, w_q, scales, biases, gs, b):
    def stock():
        return mx.quantized_matmul(x, w_q, scales, biases,
                                    transpose=True, group_size=gs, bits=b)
    return stock

def make_spqt(x, w_zz, scales_zz, biases_zz, idx):
    def spqt():
        return mlx_spqt.zigzag_qmv(x, w_zz, scales_zz, biases_zz, idx)
    return spqt

# ----- mode wrappers: per-call vs. throughput -----
N_INNER = 50    # ops per batched iter; tune so steady-state regime is reached

def latency_fn(single_op):
    return single_op            # 1 op per fn call → per-call latency

def throughput_fn(single_op, n_inner):
    def batched():
        # build N ops without evaling; time_fn's outer eval batches them
        return [single_op() for _ in range(n_inner)]
    return batched


def main():
    # build inputs; mx.eval(...) out of band so timing only covers the op
    stock = make_stock(...)
    spqt  = make_spqt(...)

    # per-call latency (msec/op = reported msec)
    time_fn(latency_fn(stock), msg="stock per-call (msec/op)")
    time_fn(latency_fn(spqt),  msg="SpQt  per-call (msec/op)")

    # batched throughput (msec/op = reported msec / N_INNER)
    time_fn(throughput_fn(stock, N_INNER), msg=f"stock throughput (msec/×{N_INNER})")
    time_fn(throughput_fn(spqt,  N_INNER), msg=f"SpQt  throughput (msec/×{N_INNER})")
```

Why a closure factory rather than passing inputs to `time_fn` via `*args`?
`time_fn` accepts args, but the throughput wrapper needs to construct the inner
list itself, which is hard to express through `*args`. Pre-binding inputs into
each `single_op` closure keeps the bench file flat and parametric.

**Picking `N_INNER`.** Two pressures:
- Big enough that command-buffer commit overhead is amortized — `max_ops_per_buffer ≈ 50` on M-series, so `N_INNER ≥ 50` ensures we cross at least one commit boundary per iter.
- Small enough that one timing iter is fast (`time_fn` runs 100 of them).

Start at 50; verify `N_INNER × 2` produces ~2× the reported msec (linear scaling = steady-state regime).

## 4. Implications for our milestones

| Milestone | Implication |
|---|---|
| **M0b** | Clone axpby scaffold; trivial registered kernel; smoke kernels for atomics, threadgroup memory, uint32 buffer. The `mx.fast.metal_kernel` checklist from earlier drafts is no longer the path — Route B is. |
| **M1/M2 correctness gate** | Use `(y_q - y_ref).abs().max() < 1e-3` (matches `test_quantized.py::test_qmv:355`), not NMSE. Same predicate for both milestones. |
| **M1 reference** (dense) | `w_q, scales, biases = mx.quantize(w, gs, b)`; `y_ref = mx.quantized_matmul(x, w_q, scales, biases, transpose=True, ...)`. No sparsity. Validates layout correctness. |
| **M2 reference** (sparse) | Mirror `test_qmv` structure: `y_ref = mx.quantized_matmul(x_sparse, w_q, scales, biases, ...)` where `x_sparse` is the activation with the sparse columns zeroed. Validates idx-walk correctness. |
| **M3 bench** | `time_fn` for both modes; closure-factory pattern (see §3). Per-call: `fn` returns one op → per-call latency. Throughput: `fn` returns list of N ops → batched; per-op = reported / N. Report both per variant. |
| **M3 baselines** | Materialize all inputs with `mx.eval(...)` before the timed region so only the qmv shows up in the measurement. |

## 5. Open questions / further look

Things to verify with a small hands-on test, in priority order:

1. **Atomic-add into output buffer from MSL.** Does the Metal compiler accept `atomic_fetch_add_explicit(...)` on a `device atomic_float*` output? Required for cross-TG row-partial reduce. (M0b smoke kernel.)
2. **`threadgroup` memory.** Standard MSL `threadgroup` qualifier inside the kernel body — should work, but worth a 30-line confirmation. (M0b smoke kernel.)
3. **`uint32` weight buffer.** `mx.array(np.random.randint(...).astype(np.uint32))` round-tripped through a kernel that does bit-unpacking — verify it Just Works. (M0b smoke kernel.)
4. **Extension build environment.** `pip install -e .` succeeds with our toolchain (CMake 3.27+, nanobind, `MLX_BUILD_METAL=ON`); `mlx_build_metallib` produces a metallib; `current_binary_dir()` + `d.get_library(...)` resolve it at runtime. (M0b scaffold.)
5. **Pipeline-state caching characterization.** First call latency vs. steady-state — quantify by how much, so we know how many warmup iters are actually needed. (M3 territory.)
6. **`mx.metal.start_capture` workflow.** Confirm it captures extension-launched dispatches the same way it captures built-in kernels — useful for verifying the wall-clock timing reflects kernel time, not framework overhead. (M3 territory.)

(1)-(4) are M0b. (5)-(6) are M3.
