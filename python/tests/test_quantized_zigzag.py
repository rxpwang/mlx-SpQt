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

# bits=8 and bits=6 added in Stage 2 (was bits=4 only in MVP).
BITS = [4, 6, 8]


def _make_sparse_idx(K, density, seed):
    """Random sorted subset of K-positions with count-prefix format."""
    n = max(1, int(K * density))
    rng = np.random.default_rng(seed=seed)
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)
    idx_np = np.insert(idx_np, 0, n)  # count prefix
    return mx.array(idx_np)


def _make_active_idx(M, density, seed):
    """Random sorted subset of M-output rows with count-prefix format."""
    n = max(1, int(M * density))
    rng = np.random.default_rng(seed=seed)
    idx_np = np.sort(rng.choice(M, n, replace=False)).astype(np.int32)
    idx_np = np.insert(idx_np, 0, n)
    return mx.array(idx_np), idx_np[1:].tolist()


class TestQuantizedZigzag(mlx_tests.MLXTestCase):
    def test_quantize_dequantize_roundtrip(self):
        """Per-element dequantize error should be bounded by the group scale.

        Mirrors test_quantized.test_quantize_dequantize: each n-bit bin has
        width = scale, so |w - dequantize(quantize(w))| <= scale.
        """
        for bits in BITS:
            for M, K in [(1024, 1024), (4096, 4096)]:
                with self.subTest(M=M, K=K, bits=bits):
                    mx.random.seed(42)
                    w = mx.random.normal(shape=(M, K)).astype(mx.float32)
                    w_q, sc, bs = quantize_zigzag(w, group_size=64, bits=bits)
                    w_dq = dequantize_zigzag(w_q, sc, bs, group_size=64, bits=bits)
                    eps = 1e-5
                    err_max = (w - w_dq).abs().max().item()
                    sc_max = sc.abs().max().item()
                    self.assertLess(err_max, sc_max + eps,
                                    f"bits={bits} err_max={err_max} > sc_max={sc_max}")

    def test_zigzag_qmv_dense(self):
        for bits in BITS:
            for M, K in SHAPES:
                with self.subTest(M=M, K=K, bits=bits):
                    mx.random.seed(42)
                    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                    x = mx.random.normal(shape=(1, K)).astype(mx.float16)

                    w_zz, scales, biases = quantize_zigzag(w, group_size=64, bits=bits)
                    y_zz = mx.zigzag_qmv_dense(x, w_zz, scales, biases, group_size=64, bits=bits)

                    sc_f32 = scales.astype(mx.float32)
                    b_f32 = biases.astype(mx.float32)
                    w_dq = dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=64, bits=bits)
                    y_ref = x.astype(mx.float32) @ w_dq.astype(mx.float32).T
                    err = (y_zz - y_ref).abs().max().item()
                    self.assertLess(err, 1e-3,
                                    f"bits={bits} M={M} K={K}: err {err:.6f} exceeds threshold")

    def test_zigzag_qmv_sparse(self):
        densities = [1.0, 0.5]
        for bits in BITS:
            for M, K in SHAPES:
                for density in densities:
                    with self.subTest(M=M, K=K, density=density, bits=bits):
                        mx.random.seed(42)
                        w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                        x = mx.random.normal(shape=(1, K)).astype(mx.float16)

                        w_zz, scales, biases = quantize_zigzag(w, group_size=64, bits=bits)
                        idx = _make_sparse_idx(K, density, seed=K * 1000 + int(density * 100))
                        y_zz = mx.zigzag_qmv_sparse(x, idx, w_zz, scales, biases, group_size=64, bits=bits)

                        sc_f32 = scales.astype(mx.float32)
                        b_f32 = biases.astype(mx.float32)
                        w_dq = dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=64, bits=bits)

                        idx_k = idx[1:]  # skip count prefix
                        y_ref = x.astype(mx.float32)[:, idx_k] @ w_dq.astype(mx.float32)[:, idx_k].T
                        err = (y_zz - y_ref).abs().max().item()
                        self.assertLess(err, 1e-3,
                                        f"bits={bits} M={M} K={K} density={density}: err {err:.6f} exceeds threshold")

    def test_zigzag_qmv_mskip(self):
        """M-skip: active rows must match dense GEMV exactly, inactive rows must be 0."""
        # Stick to a tighter shape subset — mskip's correctness is bits-orthogonal,
        # so the same shapes as the dense test would be redundant.
        mskip_shapes = [(1024, 1024), (2048, 2048), (3072, 1024), (1024, 3072)]
        densities = [1.0, 0.65, 0.35]
        for bits in BITS:
            for M, K in mskip_shapes:
                for density in densities:
                    with self.subTest(M=M, K=K, density=density, bits=bits):
                        mx.random.seed(42)
                        w = mx.random.normal(shape=(M, K)).astype(mx.float16)
                        x = mx.random.normal(shape=(1, K)).astype(mx.float16)

                        w_zz, scales, biases = quantize_zigzag(w, group_size=64, bits=bits)
                        active_idx, active_list = _make_active_idx(
                            M, density, seed=M * 1000 + int(density * 100))

                        y_mskip = mx.zigzag_qmv_mskip(
                            x, active_idx, w_zz, scales, biases, group_size=64, bits=bits)
                        y_dense = mx.zigzag_qmv_dense(
                            x, w_zz, scales, biases, group_size=64, bits=bits)
                        mx.eval(y_mskip, y_dense)

                        y_mskip_np = np.array(y_mskip)
                        y_dense_np = np.array(y_dense)

                        active_arr = np.array(active_list, dtype=np.int64)
                        diff_active = np.abs(
                            y_mskip_np[..., active_arr] - y_dense_np[..., active_arr]).max()
                        self.assertLess(diff_active, 1e-6,
                                        f"bits={bits} M={M} K={K} dens={density}: "
                                        f"active rows diverge from dense by {diff_active:.2e}")

                        mask = np.ones(M, dtype=bool)
                        mask[active_arr] = False
                        if mask.any():
                            inactive_max = np.abs(y_mskip_np[..., mask]).max()
                            self.assertEqual(inactive_max, 0.0,
                                             f"bits={bits} M={M} K={K} dens={density}: "
                                             f"inactive rows not zero (max abs={inactive_max:.2e})")


if __name__ == "__main__":
    unittest.main()