# I5 — Smoke-validate headline numbers

Goal: confirm `mx.zigzag_qmv_*` reproduces the ~1.8× speedup at
LLM-FFN shapes that the extension version produced. This is the
acceptance signal that the integration didn't regress perf.

After I5, we have end-to-end validation: integrated kernels are
correct (I4) and fast (I5). The integration milestone is done.

## Approach

Easiest path: copy the extension's
`bench_zigzag_qmv_sparse.py` to the repo root as
`bench_integration.py` and search-replace 4 imports/calls.

Total effort: ~5 minutes of edits + ~3 minutes of bench runtime.

## Step 1 — Create `bench_integration.py`

```bash
cp extensions/mlx_spqt/bench_zigzag_qmv_sparse.py bench_integration.py
```

Then edit the new file to swap extension calls for core calls:

| Find | Replace |
|---|---|
| `import mlx_spqt` | `from mlx.spqt import quantize_zigzag, dequantize_zigzag` |
| `mlx_spqt.zigzag_qmv_dense` | `mx.zigzag_qmv_dense` |
| `mlx_spqt.zigzag_qmv_sparse` | `mx.zigzag_qmv_sparse` |
| `mlx_spqt.quantize_zigzag` | `quantize_zigzag` |
| `mlx_spqt.dequantize_zigzag` | `dequantize_zigzag` |

Also fix the `_MLX_BENCHMARKS` path (since the file is no longer in
`extensions/mlx_spqt/`):

```python
_MLX_BENCHMARKS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmarks", "python")
```

(One `..` removed — file moved up one directory.)

## Step 2 — Run

```bash
python bench_integration.py
```

Expected output: same per-call μs numbers as the README headline
table, within ~5% noise. Specifically at LLM-FFN shapes:

| Shape | sparse@0.25 should beat mx.quantized_matmul by |
|---|---|
| `(4096, 11008)` | ~1.8-1.9× |
| `(11008, 4096)` | ~1.7-1.8× |
| `(4096, 16384)` | ~1.7-1.8× |

If the speedup is much lower (e.g., < 1.3×) at any of those shapes,
the integration regressed something. Most likely cause: the kernel
itself is the same (we copied source verbatim), so suspect:
- A different threadgroup geometry getting dispatched. Verify
  the `nsg=2, tg=4` defaults in the binding match what's actually
  passed.
- A new Metal pipeline cache miss on every call. Check that the
  kernel is being JIT'd / loaded once, not per-call.

## What "done" looks like

After I5 succeeds, the integrated MLX core has:
- **Correctness**: I4's pytest suite passes (7×2 dense, 7×2 sparse).
- **Performance**: I5's bench shows the headline ~1.8× speedup
  reproduces.
- **API coverage**: `mx.zigzag_qmv_dense`, `mx.zigzag_qmv_sparse`,
  `mlx.spqt.quantize_zigzag`, `mlx.spqt.dequantize_zigzag` all work.

This is the natural commit point for the integration branch.
