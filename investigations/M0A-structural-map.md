# M0A — Structural map: ggml/llama.cpp ↔ MLX

The decision-driving comparison underlying the milestone plan. Pairing principle:
**ggml ≈ mlx-core, llama.cpp ≈ mlx.nn** — don't compare across layers.
Long-form exploration of any individual row lives in the sibling `mlx-*.md` docs.

| # | Topic | ggml / llama.cpp | MLX | What we use |
|---|---|---|---|---|
| 1 | Q4 weight format | `block_q4_K` — 256-weight super-block, hierarchical super-block scale/min + 6-bit packed sub-block scales | `mx.quantize(mode="affine")` → `(packed_w: uint32, scales: fp16, biases: fp16)`; group_size 64, 8 nibbles per `uint32`, flat per-group scales | **MLX-affine.** Flat groups simplify M1's `zigzag_quantize` — no super-block hierarchy, no 6-bit-scale gymnastics. |
| 2 | Custom op registration | Add `GGML_OP_*` enum, recompile core | `mx.fast.metal_kernel` (Python-JIT) **or** C++ extension à la `examples/extensions/axpby` (registered `Primitive` + pip-installable metallib) | **C++ extension (Route B).** Registers a real `Primitive` in M0b. A future `mx.quantized_matmul(mode="affine_zigzag")` slots into the same shape. |
| 3 | Baseline GEMV kernel | `kernel_mul_mv_q4_K_f32` — 8 rows/TG, 2 SG/TG, walks K serially in 256-block chunks | `affine_qmv_fast<half, 64, 4, batch=0>` (`quantized.h:749`) — 8 rows/TG, 2 SG/TG, walks K in 512-block chunks; `qdot` / `load_vector` helpers handle the inner unpack-and-FMA | **Bench against this directly.** Per Working Principle #4, our M1 kernel mirrors its template signature + helpers; diverges only at the K-walk and the cross-TG atomic-reduce on the output. |
| 4 | Sparse-aware GEMV reference | `kernel_mul_mv_q4_K_f32_zigzag_multi_threadgroup_sparse_impl` (`llama.cpp-SpQt/ggml/src/ggml-metal/ggml-metal.metal:5867`); dense sibling at `:5702` | none — we're building it | **M1 ports the dense multi-TG variant** (`_multi_threadgroup_impl`); **M2 adds the idx-driven K-walk** on top of it. ~10-20 line diff between M1 and M2. |
| 5 | Op evaluation model | **Explicit** graph: build `ggml_cgraph` via `ggml_build_forward_expand`, dispatch with `ggml_backend_graph_compute(backend, gf)` (blocks on caller side until GPU completes) | **Implicit** lazy DAG via `array.inputs()`; `mx.eval()` materializes via DFS → tape → dispatch; one default GPU stream; async commit, blocks via synchronizer event | Both are deferred-execute models — only the graph-construction style differs. **Single-stream FIFO in our scenario** means no fence/event sync to worry about; bench timer wraps `mx.eval` after warmup. |
| 6 | Bench convention | `test_case::eval_perf()` at `tests/test-backend-ops.cpp:969` — formal op-bench runner sharing the `test_case` abstraction with correctness tests; warmup run + **adaptive `n_runs`** sized to target FLOPs (~100 GFLOPs on GPU) or memory (~32 GB on GPU); registered via `make_test_cases_perf()` (`:5316`); invoked as `test-backend-ops perf`. Plus standalone SpQt benches: `tests/bench-{sparse-gemv,decode-breakdown,sparse-indexing}.cpp`. | `time_fn` from `benchmarks/python/time_utils.py:8` — standalone Python utility (not test-integrated); fixed 5 warmup + 100 timed iters; mean ms per `fn` call; `mx.eval`-only wait; no adaptive iter count. | **Vendor `time_fn` into `tests/`.** Closure-factory wrappers expose **per-call latency** (`fn` returns 1 op) vs. **batched throughput** (`fn` returns N un-eval'd ops; per-op = reported / N) from one single-op closure. We accept the lower-ceremony MLX-side convention; adaptive `n_runs` à la ggml is an enhancement worth picking up if M3 perf gating becomes shape-sensitive. |
| 7 | Correctness test convention | `test-backend-ops` (CPU-vs-GPU compare, NMSE < 5e-4 threshold) | `unittest` + `(y - y_ref).abs().max() < 1e-3` (max-abs-diff predicate at `python/tests/test_quantized.py::test_qmv:355`) | **Mirror `test_qmv` structure.** M1 gate: dense max-abs-diff < 1e-3 against stock `mx.quantized_matmul`. M2 gate: same predicate at 50% sparsity, with sparse cols of `x` zeroed in the reference. |

---

## Why these seven rows, and what they each gate

- **Row 1** decides what `zigzag_quantize` operates on (M1).
- **Row 2** decides our extension scaffolding (M0b) and the kernel's API surface (M2).
- **Row 3** decides what we benchmark against (M3) and what our kernel structurally mimics (M1, per Working Principle #4).
- **Row 4** decides what we port from (M1, M2).
- **Row 5** decides bench-timer wiring (M3) and confirms why our setup is FIFO-clean.
- **Row 6** decides bench code shape (M3).
- **Row 7** decides correctness-gate code shape (M1, M2).

Anything not on this table is FYI for the deeper investigations or for the talk; it's not driving an M1–M4 decision.
