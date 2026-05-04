# I4 — Port tests to MLX pytest layout

Goal: a unittest-style test file in `python/tests/` that mirrors
the extension's correctness tests, using `mx.zigzag_qmv_*` and
`mlx.spqt.*` instead of `mlx_spqt.*`.

After I4, `python -m pytest python/tests/test_quantized_zigzag.py`
runs the same 7-shape × 2-density correctness coverage as the
extension's tests.

## File to create

`python/tests/test_quantized_zigzag.py` — single test file with
three test methods.

## Style: `mlx_tests.MLXTestCase`

MLX's test pattern (see `python/tests/test_quantized.py:10`):

```python
import unittest
import mlx.core as mx
import mlx_tests

class TestQuantizedZigzag(mlx_tests.MLXTestCase):
    def test_xxx(self):
        ...

if __name__ == "__main__":
    unittest.main()
```

## Step 1 — Create `python/tests/test_quantized_zigzag.py`

```python
# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np
from mlx.spqt import quantize_zigzag, dequantize_zigzag


SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (4096, 11008),
    (11008, 4096),
    (4096, 16384),
]


def _make_sparse_idx(K, density, seed):
    """Random sorted subset of K-positions with count-prefix format."""
    n = max(1, int(K * density))
    rng = np.random.default_rng(seed=seed)
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)
    idx_np = np.insert(idx_np, 0, n)  # count prefix
    return mx.array(idx_np)


class TestQuantizedZigzag(mlx_tests.MLXTestCase):
    def test_quantize_dequantize_roundtrip(self):
        """quantize_zigzag → dequantize_zigzag should be bit-stable
        (round-trip a quantized array twice and get the same result)."""
        for M, K in [(1024, 1024), (4096, 4096)]:
            with self.subTest(M=M, K=K):
                mx.random.seed(42)
                w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                w_q, sc, bs = quantize_zigzag(w)
                w_dq = dequantize_zigzag(w_q, sc, bs)
                # second round-trip should be identical
                w_q2, sc2, bs2 = quantize_zigzag(w_dq.astype(mx.float16))
                self.assertTrue(mx.array_equal(w_q, w_q2))

    def test_zigzag_qmv_dense(self):
        """Dense GEMV should match the fp32 reference within 1e-3."""
        for M, K in SHAPES:
            with self.subTest(M=M, K=K):
                mx.random.seed(42)
                w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                x = mx.random.normal(shape=(1, K)).astype(mx.float16)
                w_q, sc, bs = quantize_zigzag(w)

                y = mx.zigzag_qmv_dense(x, w_q, sc, bs)
                self.assertEqual(y.shape, (1, M))
                self.assertEqual(y.dtype, mx.float32)

                # fp32 reference: dequant + matmul on the full K
                w_dq = dequantize_zigzag(w_q, sc.astype(mx.float32),
                                          bs.astype(mx.float32))
                y_ref = x.astype(mx.float32) @ w_dq.astype(mx.float32).T
                err = (y - y_ref).abs().max().item()
                self.assertLess(err, 1e-3, f"err={err}")

    def test_zigzag_qmv_sparse(self):
        """Sparse GEMV should match `x[:,idx] @ w_dq[:,idx].T` within 1e-3."""
        for M, K in SHAPES:
            for d in [1.0, 0.5]:
                with self.subTest(M=M, K=K, density=d):
                    mx.random.seed(42)
                    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                    x = mx.random.normal(shape=(1, K)).astype(mx.float16)
                    w_q, sc, bs = quantize_zigzag(w)
                    idx = _make_sparse_idx(K, d, seed=K * 1000 + int(K * d))

                    y = mx.zigzag_qmv_sparse(x, idx, w_q, sc, bs)
                    self.assertEqual(y.shape, (1, M))
                    self.assertEqual(y.dtype, mx.float32)

                    w_dq = dequantize_zigzag(w_q, sc.astype(mx.float32),
                                              bs.astype(mx.float32))
                    idx_k = idx[1:]
                    y_ref = (x.astype(mx.float32)[:, idx_k]
                             @ w_dq.astype(mx.float32)[:, idx_k].T)
                    err = (y - y_ref).abs().max().item()
                    self.assertLess(err, 1e-3, f"err={err} M={M} K={K} d={d}")


if __name__ == "__main__":
    unittest.main()
```

## Step 2 — Run

```bash
python -m unittest python.tests.test_quantized_zigzag -v
```

Or with pytest:

```bash
python -m pytest python/tests/test_quantized_zigzag.py -v
```

Expected: all three tests pass. The full suite covers the same
shapes the existing extension tests cover (7 LLM-relevant shapes
× 2 densities for sparse).

## Pitfalls

1. **`mlx_tests` import** — that module is `python/tests/mlx_tests.py`,
   provides `MLXTestCase` with stream / device fixtures. It's
   importable from any test file in `python/tests/` because pytest
   runs with that dir on the path.

2. **`mx.array_equal` for the bit-stable round-trip** — using
   element-wise `==` would still work but `array_equal` is the
   idiomatic check for "all elements identical."

3. **Test runtime** — sparse covers 7×2 = 14 cases at large shapes;
   each takes a few seconds for the kernel + reference matmul. Full
   test run is ~1-2 minutes. If too slow for iterative dev, run
   `test_zigzag_qmv_dense` alone or trim SHAPES to `(1024,1024)` +
   `(4096,11008)` for spot-check.

4. **`subTest`** — wraps each shape so a single failure doesn't
   abort the rest. Get the failing case in the output without
   re-running.
