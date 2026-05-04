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
        """Per-element dequantize error should be bounded by the group scale.

        Mirrors test_quantized.test_quantize_dequantize: each 4-bit bin has
        width = scale, so |w - dequantize(quantize(w))| <= scale.
        """
        for M, K in [(1024, 1024), (4096, 4096)]:
            with self.subTest(M=M, K=K):
                mx.random.seed(42)
                w = mx.random.normal(shape=(M, K)).astype(mx.float32)
                w_q, sc, bs = quantize_zigzag(w)
                w_dq = dequantize_zigzag(w_q, sc, bs)
                eps = 1e-5
                err_max = (w - w_dq).abs().max().item()
                sc_max = sc.abs().max().item()
                self.assertLess(err_max, sc_max + eps,
                                f"err_max={err_max} > sc_max={sc_max}")

    def test_zigzag_qmv_dense(self):
        for M, K in SHAPES:
            with self.subTest(M=M, K=K):
                mx.random.seed(42)
                w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                x = mx.random.normal(shape=(1, K)).astype(mx.float16)

                w_zz, scales, biases = quantize_zigzag(w)
                y_zz = mx.zigzag_qmv_dense(x, w_zz, scales, biases)

                sc_f32 = scales.astype(mx.float32)
                b_f32  = biases.astype(mx.float32)
                w_dq   = dequantize_zigzag(w_zz, sc_f32, b_f32)
                y_ref  = x.astype(mx.float32) @ w_dq.astype(mx.float32).T
                err = (y_zz - y_ref).abs().max().item()
                self.assertLess(err, 1e-3, f"M={M} K={K}: err {err:.6f} exceeds threshold")

    def test_zigzag_qmv_sparse(self):
        densities = [1.0, 0.5]
        for M, K in SHAPES:
            for density in densities:
                with self.subTest(M=M, K=K, density=density):
                    mx.random.seed(42)
                    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                    x = mx.random.normal(shape=(1, K)).astype(mx.float16)

                    w_zz, scales, biases = quantize_zigzag(w)
                    idx = _make_sparse_idx(K, density, seed=K * 1000 + int(density * 100))
                    y_zz = mx.zigzag_qmv_sparse(x, idx, w_zz, scales, biases)

                    sc_f32 = scales.astype(mx.float32)
                    b_f32  = biases.astype(mx.float32)
                    w_dq   = dequantize_zigzag(w_zz, sc_f32, b_f32)

                    idx_k = idx[1:]  # skip count prefix
                    y_ref  = x.astype(mx.float32)[:, idx_k] @ w_dq.astype(mx.float32)[:, idx_k].T
                    err = (y_zz - y_ref).abs().max().item()
                    self.assertLess(err, 1e-3, f"M={M} K={K} density={density}: err {err:.6f} exceeds threshold")

if __name__ == "__main__":
    unittest.main()