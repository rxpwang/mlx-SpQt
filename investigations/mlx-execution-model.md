# MLX execution model: lazy graph, eval, and Metal dispatch

A free-exploration dive into what actually happens between `mx.quantized_matmul(...)` (the Python call) and the GPU finishing the kernel. Written so future-you can reconstruct the model without re-reading `transforms.cpp` and `eval.cpp` end to end.

All file/line references are against `rxpwang/spqt` at `e8ebdebe` (vanilla upstream MLX, post-0.31.2).

## TL;DR

- MLX is **lazy**: ops construct DAG nodes; nothing computes until something forces materialization.
- The "graph" isn't a separate object — it's implicit in `array.inputs()` and `array.primitive_ptr()` chains.
- `mx.eval(y)` is a **3-phase algorithm** inside `eval_impl`: DFS for ancestors → build a topo-ordered tape → pop the tape and dispatch to streams.
- A **stream** owns a Metal command queue. Within a stream, kernels are encoded into rotating **command buffers** (~50 ops each) and submitted to the GPU asynchronously.
- For our single-GPU microbench: one stream, one queue, kernels run FIFO.
- `commit()` ≠ compute. `commit()` enqueues a buffer; the GPU runs it asynchronously; eval blocks on a synchronizer event at the end.

---

## 1. The lazy graph is not a graph object

MLX has no `Graph` class, no global registry, no scheduler queue holding nodes. The graph **exists implicitly** in the linkage between `array`s.

Each `array` carries (`mlx/array.h:205-211, 268-281`):
- `array_desc_->primitive` — `shared_ptr<Primitive>`. The op that produces this array. `nullptr` for leaves (arrays with their own data, not derived from other arrays).
- `array_desc_->inputs` — vector of `array`s the primitive depends on.
- `array_desc_->siblings` — other arrays produced by the same primitive (multi-output ops).
- `Status: {unscheduled, scheduled, evaluated}` — drives whether eval visits the node.

So "the graph" is whatever you can reach by walking `array.inputs()` recursively. There's no top-level data structure to inspect.

---

## 2. Phase A: graph construction (when Python returns)

```python
y = mx.quantized_matmul(x, w_q, scales, biases, ...)
```

The chain (no compute happens):

```
mx.quantized_matmul(...)                         # Python
  → mx::quantized_matmul                         # python/src/ops.cpp:4312 (binding)
    → mlx::core::quantized_matmul               # mlx/ops.cpp:4477
      ├─ shape validation
      ├─ new QuantizedMatmul primitive          # mlx/ops.cpp:4524
      └─ array::make_arrays(out_shape, dtype, prim, inputs)
                                                 # → returns an unscheduled array
```

Result: a new `array y` with `primitive=QuantizedMatmul`, `inputs={x, w_q, scales, biases}`, no allocated buffer, `Status::unscheduled`. **Python returns. The GPU has not been touched.**

---

## 3. Phase B: eval — DFS, tape, dispatch

`mlx/transforms.cpp:74-313` (`eval_impl`).

When you call `mx.eval(y)` (or anything that materializes — `print`, `np.array`, etc.):

### 3.1 — Phase 1: DFS for unevaluated ancestors

A `std::stack` walks back from `y` via `inputs()`, only recursing into nodes whose `Status == unscheduled`. A `cache` map records each node's fan-out degree (how many descendants depend on it) — this is what gives DAG semantics: a node referenced by N descendants is computed *once*, not N times.

`transforms.cpp:108-172`. Outputs of this phase:
- A set of nodes that need computing.
- A `needs_fence` map for cross-stream dependencies.

### 3.2 — Phase 2: build the tape in topological order

A `std::deque<array> tape` is filled (`transforms.cpp:174-218`) using degree counts from phase 1. Nodes appear *after* their inputs and *before* their consumers. There's a configurable BFS width limit (`env::bfs_max_width()`) to bound peak memory.

### 3.3 — Phase 3: pop the tape and dispatch

`transforms.cpp:222-301`. The main loop. Per popped node:

1. **Cross-stream sync.** Walk inputs; insert fence/event waits for any input produced on a different stream (`:242-256`). Within-stream dependencies are already serialized by Metal command-buffer ordering — no fence needed.
2. **Encode.** Call `gpu::eval(arr)` (or `cpu::eval` for CPU primitives). This is what *encodes* the work; it does not run on the GPU yet. See §4.
3. **Update fences.** If `arr` is an input to something on another stream, store its fence value (`:280-288`).
4. **Status.** `arr.set_status(evaluated)`. If not a tracer, `arr.detach()` — drops the primitive shared_ptr and the inputs, so the graph behind `arr` is GC'd.
5. **Backpressure check.** If `n_active_tasks > MAX_ACTIVE_TASKS` or memory exceeds limit, force-commit open streams and `wait_for_one()` until a buffer drains (`:264-278`).

After tape is empty, the **epilogue** (`:303-310`):
- Signal events on each open stream (so async-eval consumers can wait).
- Call `gpu::finalize(s)` for each GPU stream — force-commits any partial command buffer.

The eval call returns the synchronizer array; the synchronous wrapper waits on the synchronizer's event before returning to Python.

---

## 4. Phase 3 zoom: how `gpu::eval(arr)` encodes and commits

`mlx/backend/metal/eval.cpp:30-75`.

```cpp
void eval(array& arr) {
    auto pool = metal::new_scoped_memory_pool();
    auto s = arr.primitive().stream();
    auto& encoder = metal::get_command_encoder(s);          // MLX's wrapper; persistent per stream
    auto* command_buffer = encoder.get_command_buffer();    // current MTLCommandBuffer

    arr.primitive().eval_gpu(arr.inputs(), arr.outputs());  // ★ writes kernel commands into the buffer

    // Capture input buffers so they outlive the GPU work
    std::unordered_set<std::shared_ptr<array::Data>> buffers = ...;

    if (encoder.needs_commit()) {                           // buffer too full?
        encoder.end_encoding();
        scheduler::notify_new_task(s);                      // bump in-flight counter
        command_buffer->addCompletedHandler([s, buffers](auto* cbuf) {
            scheduler::notify_task_completion(s);           // GPU side: decrement
            check_error(cbuf);
        });
        encoder.commit();                                   // submit to queue (async!)
    } else {
        // Encoder still has room; just attach error-check handler
        command_buffer->addCompletedHandler([buffers](auto* cbuf) { check_error(cbuf); });
    }
}
```

`primitive.eval_gpu(...)` is what we already traced for our QuantizedMatmul — it goes `dispatch_qmv` → `qmv` → `dispatch_threadgroups`, and each `dispatch_threadgroups` call **encodes a kernel dispatch into the current `MTLComputeCommandEncoder`**.

### Command-buffer batching

`mlx/backend/metal/device.cpp:445-448`:

```cpp
bool CommandEncoder::needs_commit() const {
    auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
    return (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb);
}
```

Defaults are device-dependent (`device.cpp:501-521`): 20 ops on M1, 40 on M2, 50 on M3/M4. Many primitives normally share one command buffer; the buffer is committed when full or at `finalize`. This amortizes per-commit overhead.

### `gpu::finalize`

`mlx/backend/metal/eval.cpp:77-84`. Called once per stream at the end of phase 3. Force-commits any uncommitted command buffer regardless of fill level. Without this, the last few kernels of an eval might sit in an open encoder forever.

---

## 5. The Metal layer cake

```
MTLDevice                   ← the GPU
  ↓
MTLCommandQueue             ← one per Stream
  ↓
MTLCommandBuffer            ← the unit of submission. Holds encoded commands.
  ↓
MTLComputeCommandEncoder    ← writes kernel dispatches into the parent buffer
```

Two vocab points:

- **The buffer is what gets submitted; the encoder is what writes into it.** End the encoder before committing the buffer.
- **`commit()` doesn't compute.** It returns immediately. The GPU dequeues the buffer asynchronously and runs encoded kernels in order. The completion handler runs (on a CPU thread) when the GPU finishes.

MLX's `CommandEncoder` (`device.h:84-180`) is a wrapper that bundles the current buffer + current compute encoder + bookkeeping (`buffer_ops_`, `buffer_sizes_`). One per stream, persistent. Internally it rotates through Metal command buffers.

---

## 6. Streams in our scenario

For a single-GPU microbench with no manual stream management:

- **One default stream.** `default_stream(default_device())` returns the same `Stream` object every time. Every primitive constructed without an explicit `stream=` resolves to this one (`transforms.cpp:33`).
- **All primitives in the graph point to that stream.** Cross-stream sync logic in phase 3 finds `consumer.stream() == producer.stream()` for every dependency → no fences inserted, no events waited.
- **Pure FIFO.** One `MTLCommandQueue`, kernels run in encode order.

Two caveats that don't bite us in the MVP but are worth knowing:

1. **CPU fallback inserts a CPU stream.** If any op in the graph isn't GPU-implemented and runs on CPU, that's a *different* `Stream`, and a fence is inserted between them. Worth checking with `mx.metal.start_capture` if eval feels mysteriously slow.
2. **`Device` ≠ `Stream`.** `mx.gpu` is a `Device`; its default stream is what's actually used. `to_stream(...)` resolves either form.

---

## 7. Status state machine on each array

```
[constructed by op]   ────▶  unscheduled        # has primitive, no data
[during phase 3]      ────▶  scheduled          # encoded; GPU may not be done
[after stream signal] ────▶  evaluated          # data buffer is materialized
```

`array.is_available()` checks `evaluated` and that the underlying buffer is ready. Print/numpy/etc. trigger eval if not yet `evaluated`.

---

## 8. The synchronizer: how `eval` blocks until done

`transforms.cpp:53, 89-101, 313`. The synchronization mechanism:

1. A `Synchronizer` primitive is constructed with all `outputs` as its inputs.
2. An `Event` is attached to it, `set_value(1)` (pending state).
3. The synchronizer is added to the tape so it's processed in phase 3.
4. After the epilogue's `gpu::finalize(s)`, the GPU eventually completes the last buffer and signals the event.
5. The synchronous wrapper of `eval()` waits on that event before returning.

`async_eval` uses the same machinery but returns *without* waiting. The event stays attached to the outputs, so a later op that consumes them will wait on it (the cross-eval-call event handling in step 1 of phase 3).

---

## 9. What this means for SpQt benchmarking

1. **Construction is free.** `y = my_zigzag_qmv(...)` doesn't run anything. The bench timer must wrap `mx.eval(y)` (or `mx.synchronize()`), not the call.

2. **First eval is more expensive.** Pipeline state caching, metallib loading, scratch allocation. Always do ≥1 warm-up before timing.

3. **`mx.synchronize()` is the cleanest "wait for GPU".** Useful at start of bench (drain prior work) and end of timed region.

4. **Per-call vs batched timing.** Two distinct numbers, both useful:
   - **Per-call latency:** `eval` inside the loop. Each kernel gets its own command buffer (1 op << max_ops, finalize commits it). Includes commit + GPU run + completion overhead per iteration. This is what's seen in real decode workloads.
   - **Batched throughput:** `outs = [qmv(...) for _ in range(N)]; mx.eval(outs)`. ~N ops share 1-2 command buffers (depending on `max_ops_per_buffer`). Per-iteration commit overhead is amortized. This shows steady-state kernel-only cost.

5. **The eval timer includes more than the kernel.** Roughly: `encode (~µs CPU) + commit (~µs CPU) + GPU dequeue + GPU kernel + completion handler return`. To isolate the on-device kernel time, use `mx.metal.start_capture` and inspect with Xcode Instruments.

6. **Commit batching is not in our control.** If we want a clean "one kernel per buffer" comparison, eval inside the loop. If we want amortized throughput, batch and eval once.

7. **Single-stream means no cross-stream noise.** Our perf comparison between SpQt-MLX qmv and stock `affine_qmv_fast` will not see fence/event stalls. Apples-to-apples.
