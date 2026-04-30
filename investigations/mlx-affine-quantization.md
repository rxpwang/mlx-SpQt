# MLX-affine quantization on Metal

A free-exploration dive ahead of M0a, to understand what we're porting *into* and what we're racing *against*. All file/line references are against `rxpwang/spqt` at `e8ebdebe` (vanilla upstream MLX, post-0.31.2).

## TL;DR

- MLX-Metal supports 4 quantization "modes": `affine`, `mxfp4`, `nvfp4`, `mxfp8`. Only `affine` is structurally analogous to ggml K-quants and is the path we want.
- `affine` is **MLX's original quantization** (PR #205, Dec 2023). The label was added retro-actively in PR #2499 (Aug 2025) when MX/NV-spec FP modes were introduced. It's the default everywhere and by far the most-tuned kernel family in the codebase.
- Layout: per-output-channel × per-group-along-K asymmetric linear quantization. Same family as ggml Q4_K, GPTQ, bitsandbytes 4-bit. Mathematically `w_dequant = scale * q + bias`.
- For our shape — single-sequence decode, M=1, K=4096, N=4096, fp16 acts, gs=64, b=4 — exactly **one kernel** is selected: **`affine_qmv_fast_float16_t_gs_64_b_4_batch_0`**. That's the baseline for the M3 perf gate.

## 1. Modes

| Mode | Group size | Bits | Has bias | Defined in |
|---|---|---|---|---|
| **`affine`** | **32, 64, 128** | **2, 3, 4, 5, 6, 8** | yes | `kernels/quantized.{h,metal}` |
| `mxfp4` | 32 (fixed) | 4 (fixed) | no (E8M0 scale only) | `kernels/fp_quantized.{h,metal}` |
| `nvfp4` | 16 (fixed) | 4 (fixed) | no (E4M3 scale only) | same |
| `mxfp8` | 32 (fixed) | 8 (fixed) | no (E8M0 scale only) | same |

Activation dtype: `float32`, `float16`, `bfloat16` for all four modes.

The FP modes (`mxfp4`/`nvfp4`/`mxfp8`) are MX/NV-spec block-floating-point. They're recent additions for ecosystem compatibility (running NVIDIA-pipeline NVFP4 weights, etc.), not because they outperform affine in the general case.

## 2. Why "affine"

**Origin (`git log` traces):**

- First commit: `dfa9f4bc "An initial quantized matmul implementation (#205)"`, Angelos Katharopoulos, **Dec 2023** — at MLX's birth. This was *the* quantization scheme; there was no `mode` parameter.
- Name introduced: `70560b6b "Add mode parameter for quantization (#2499)"`, Awni Hannun, **Aug 2025** — retroactive label when MX/NV FP modes were added.

**Math (`mlx/backend/cpu/quantized.cpp:131`):**

```
w_dequant[i] = scale[g] * q[i] + bias[g]    where g = i // group_size
```

This is plain per-group asymmetric linear quantization — same family as:

- **ggml Q4_K** (super-block scale + per-sub-block 6-bit scale + min)
- **GPTQ** (per-group scale + zero-point)
- **bitsandbytes Linear8bit/4bit**

What MLX calls `bias` is what Q4_K calls `min`, up to a sign convention.

**Popularity:**

- `mode="affine"` is the default in `mx.quantize`, `mx.quantized_matmul`, `mx.dequantize`, `nn.quantize` (`mlx/ops.h:1481, 1489, 1500`; `python/mlx/nn/layers/quantized.py:13`).
- Most-tuned kernel by a wide margin: ~10 distinct PRs over 2023-2025 vs. one introductory PR for the FP modes.
- HuggingFace MLX model ports, mlx-lm conversions, and `mlx-examples` repos all use affine Q4 by default.

## 3. Layout

`mx.quantize(w, group_size, bits, mode="affine")` with `w.shape = (..., M, K)` returns three arrays (`mlx/ops.cpp:4789-4798`):

| Array | Shape | dtype |
|---|---|---|
| `packed_w` | `(..., M, K * bits / 32)` | `uint32` |
| `scales` | `(..., M, K / group_size)` | same as input |
| `biases` | `(..., M, K / group_size)` | same as input |

**Constraints (`ops.cpp:4738, 4745, 4932`):** `group_size ∈ {32, 64, 128}`, `bits ∈ {2,3,4,5,6,8}`, `K % group_size == 0`.

**Grouping rule:** groups run along the **last axis (K), independently per row of M**. Each output channel has its own scales/biases — nothing shared across rows.

**Bit packing for `bits=4` (`common/quantized.h:5-12`):**
- `pack_factor = 32 / 4 = 8` weights per `uint32`.
- `bytes_per_pack = 4` (one uint32).
- Little-endian within a `uint32`: weights `[w0..w7]` at bit positions `[0:4, 4:8, ..., 28:32]` (visible at `cpu/quantized.cpp:138-143`).
- (Irregular widths 3/5/6 carry partial bits across pack boundaries — not relevant for Q4.)

**Concrete example: `w.shape = (4096, 4096)`, `gs=64`, `bits=4`:**

| Tensor | Shape | Bytes |
|---|---|---|
| `packed_w` | `(4096, 512)` `uint32` | 8 MB |
| `scales` (fp16) | `(4096, 64)` `fp16` | 512 KB |
| `biases` (fp16) | `(4096, 64)` `fp16` | 512 KB |
| Total | | **9 MB** |

Effective rate: **4.5 bits/weight** — coincidentally identical to GGUF Q4_K.

**Memory walk for one row:**

```
Row i (4096 K-positions):
┌──────────────────────────────────────────────────────────────────────┐
│  Group 0 (K=0..63)  │  Group 1 (K=64..127)  │ ... │  Group 63 (K=4032..4095) │
│  ┌──────────────┐   │  ┌──────────────┐     │     │  ┌──────────────┐       │
│  │ 8 × uint32   │   │  │ 8 × uint32   │     │ ... │  │ 8 × uint32   │       │
│  │ = 64 nibbles │   │  │ = 64 nibbles │     │     │  │ = 64 nibbles │       │
│  └──────────────┘   │  └──────────────┘     │     │  └──────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
   scale[i, 0]            scale[i, 1]                   scale[i, 63]
   bias[i, 0]             bias[i, 1]                    bias[i, 63]
```

2 KB packed + 64 scales + 64 biases per row = 2.25 KB total per row.

## 4. Comparison with ggml Q4_K

|  | MLX-affine (gs=64, b=4) | ggml Q4_K |
|---|---|---|
| Group size (weights/group) | **64** | **32** sub-block; 256 super-block |
| Group structure | Flat (one level) | Hierarchical (super-block → 8 sub-blocks) |
| Scale encoding | fp16 per group | 16-bit super-block + 6-bit packed sub-block |
| Offset encoding | fp16 `bias` per group | 16-bit super-block min + 6-bit packed sub-block min |
| Dequant math | `w = s * q + b` | `w = s_sb * (q + min_sb)` after super-block decompression |
| Bytes per 64 weights | 32 + 4 = 36 B → 4.5 bpw | (144/256) × 64 = 36 B → 4.5 bpw |
| Cross-row sharing | none | none |

**Same family, different encoding sophistication.** Q4_K spends its bits on a hierarchical scale-of-scales to compress metadata; MLX-affine just stores per-group scales in fp16. Same average bitrate falls out coincidentally.

The flat-grouped form actually **simplifies the SpQt port** — no super-block / sub-block / 6-bit scale gymnastics in our `zigzag_quantize` step.

## 5. Kernel landscape

The catalog is best read by first decoding the naming convention. From `kernels/quantized.metal:88-138`:

### Naming convention

| Token | Meaning |
|---|---|
| `q` | quantized W operand |
| `m` | matrix (multi-row) |
| `v` | vector (single row) |
| `_n` / `_t` | W not / is transposed |
| `_fast` | fast path with alignment requirements |
| `_quad` | tiny inner-dim specialization (D ∈ {64, 128}) |
| `_split_k` / `_splitk` | split-K reduction for large K |
| `_alN_true` / `_alN_false` | N divides the tile width or not |
| `_batch_0` / `_batch_1` | leading batch dims absent / present |
| `gather_…` | MoE-style index lookup on weights (LHS gather) |
| `…_rhs` | gather on activations instead of weights |
| `_nax` | Apple NAX backend (M5/A19 Pro+, K%64==0) |

Example: `affine_qmv_fast_float16_t_gs_64_b_4_batch_0` = affine × matrix-vector GEMV × fast path × fp16 acts × group_size 64 × 4-bit weights × no batch dim.

### Full catalog (per `(type, group_size, bits)`)

| Family | Templates | Variants | # |
|---|---|---|---|
| Format conversion | `affine_quantize`, `affine_dequantize` | – | 2 |
| GEMV | `affine_qmv_fast`, `affine_qmv` | × batched ∈ {0,1} | 4 |
| GEMV (small K) | `affine_qmv_quad` | × D ∈ {64,128} × batched | 4 |
| VMM | `affine_qvm` | × batched | 2 |
| VMM (split-K) | `affine_qvm_split_k` | × split ∈ {8,32} | 2 |
| GEMM | `affine_qmm_n` | × batched | 2 |
| GEMM (transposed) | `affine_qmm_t` | × aligned × batched | 4 |
| GEMM (split-K) | `affine_qmm_t_splitk` | × aligned | 2 |
| Gather GEMV | `affine_gather_qmv_fast`, `affine_gather_qmv`, `affine_gather_qvm` | – | 3 |
| Gather GEMM | `affine_gather_qmm_n`, `affine_gather_qmm_t` | × aligned (for `_t`) | 3 |
| RHS-gather GEMM | `affine_gather_qmm_rhs_nt`, `affine_gather_qmm_rhs_nn` | – | 2 |

≈ **30 specializations per `(type, gs, bits)` tuple** × 3 types × 3 group sizes × 6 bit widths = **~1620 standard-path kernel binaries** for affine alone.

The NAX backend adds ~5 more templates for `qmm_*_nax` and `gather_qmm_*_nax` (~270 binaries), but **NAX has no GEMV** — `quantized_nax.metal` only instantiates matrix-matrix variants, gated by `metal::is_nax_available() && transpose && K%64==0` (`quantized.cpp:885`, `:694-697`).

## 6. GEMV subset (the only family we care about)

For decode-time matmul (M=1, no MoE, no NAX), the catalog reduces to **3 kernel templates**:

| Template | Triggered when | TG geometry | Variants |
|---|---|---|---|
| `affine_qmv_fast` | `N % 8 == 0 && K % 512 == 0` | 2 SG/TG, 4 rows/SG = **8 rows/TG** | × batched = 2 |
| `affine_qmv` | general — fast's alignment fails | same TG geometry, half throughput | × batched = 2 |
| `affine_qmv_quad` | `K ∈ {64, 128}` AND power-of-2 bits | 1 SG/TG split into 8 quadgroups, 8 rows/quadgroup = **64 rows/TG** | × D × batched = 4 |

= 8 specializations × (3 types × 3 gs × 6 bits) = **432 GEMV kernel binaries**.

### Selection rule

`mlx/backend/metal/quantized.cpp:1365-1385`:

```c
void dispatch_qmv(...) {
  // route to qmv_quad when K is small and fully-unrollable
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(...);
    return;
  }
  qmv(...);    // qmv() then picks fast vs slow internally
}
```

Inside `qmv()` at `quantized.cpp:259`:

```c
bool fast = N % bn == 0 && K % 512 == 0;       // bn = 8
```

### `qmv_fast` geometry (`quantized.h:750-814`)

For `bits=4`, `gs=64`:

- `num_simdgroups = 2`, `results_per_simdgroup = 4` → **8 output rows per TG**.
- `pack_factor = 8`, `packs_per_thread = 2` → 16 weights/thread per K-step.
- `block_size = values_per_thread × SIMD_SIZE = 16 × 32 = 512` K-positions per outer iter.
- Each simdgroup walks K in 512-wide blocks; per block, every lane:
  1. Loads 16 contiguous `x` values into registers.
  2. **Reuses that strip across all 4 of its assigned rows** (one `qdot` per row).
  3. `simd_sum`-reduces partials and writes 4 outputs from lane 0.

This is exactly ggml's K-quants pattern (`N_R0_Q4_K = 4`, `N_SG_Q4_K = 2` → 8 rows/TG): **load `x` once, FMA into multiple W rows**, amortizing activation bandwidth.

Grid (`quantized.cpp:251-254`):

```
bn = 8;
group_dims = (32, 2, 1)    // 64 threads = 2 simdgroups
grid_dims  = (M, ceil(N/8), B)
```

### `qmv_quad` geometry (`quantized.h:693-747`)

For `D ∈ {64, 128}`, the kernel **fully unrolls K** because `D` is a compile-time constant:

- `QUAD_SIZE = 4`, `quads_per_simd = 32/4 = 8`.
- Each quadgroup (4 threads) handles `results_per_quadgroup = 8` output rows.
- Each thread loads `D / QUAD_SIZE` weights = 16 (D=64) or 32 (D=128) — entire K, no outer loop.
- Reduction is `quad_sum` (4-way) instead of `simd_sum`.
- 1 simdgroup per TG, 8 quadgroups per simdgroup → **64 output rows per TG**.

Grid (`quantized.cpp:193-198`):

```
bn = quads_per_simd × results_per_quadgroup = 64
group_dims = (32, 1, 1)    // 1 simdgroup per TG
grid_dims  = (M, ceil(N/64), B)
```

### The `batch_{0,1}` flag

Compile-time bool template parameter. From `quantized.cpp:269`:

```c
batched = (B > 1)    where B = out.size() / M / N
```

The kernel wrapper at `quantized.h:1515`:

```c
if (batched) {
    adjust_matrix_offsets<T>(...);    // walks x_shape/strides, w_shape/strides per leading dim
}
qmv_fast_impl<...>(...);              // same inner kernel either way
```

Two binaries because the branch is on a compile-time constant — keeps the inner loop's register pressure clean for the no-batch case (the dominant case for decode).

`batch_1` fires when there are leading dims beyond `(M, N)` — e.g. multi-stream decode, beam search, batched matmul. For our microbench (`x: (1, K)`, `w: (N, K)`, `out: (1, N)`): `B = 1` → **`batch_0`**.

## 7. The baseline for our MVP

For `M=1, K=4096, N=4096`, fp16 acts, `gs=64`, `bits=4`:

| Selector | Value | Kernel branch |
|---|---|---|
| K ∈ {64, 128}? | no | not `qmv_quad` |
| K % 512 == 0? | yes (4096 = 8 × 512) | fast path eligible |
| N % 8 == 0? | yes (4096 = 512 × 8) | fast path eligible |
| B > 1? | no | `batch_0` |

→ **`affine_qmv_fast_float16_t_gs_64_b_4_batch_0`**

This is the kernel SpQt-MLX must break-even against in M3.

## 8. Implications for the SpQt port

1. **Path 2 (MLX-affine) is structurally equivalent to the paper's Q4_K target.** Same per-output-channel × group-along-K layout, same per-group asymmetric scale+offset math. SpQt's zigzag idea ports without conceptual translation.

2. **The flat group structure simplifies the quantize step.** Q4_K's super-block + sub-block + 6-bit-scale hierarchy is collapsed in MLX-affine to one fp16 scale per 64 weights. Our `zigzag_quantize` does fp-rearrange + quantize as one operation against this flatter format; llama.cpp-SpQt's reference does fp-rearrange (`rearrange_tensor_zigzag`) followed by the standard Q4_K quantizer (with its 6-bit-scale gymnastics).

3. **The baseline is honest.** `qmv_fast` is the production GEMV most MLX-quantized LLMs hit at decode. Two-plus years of optimization PRs have gone into it. Break-even is a meaningful claim, not a synthetic comparison.

4. **The threadgroup geometry to beat is "8 rows × K-serial".** SpQt's pitch is to flip orientation: tile K across TGs (with row-local partials reduced via atomics), not tile rows across TGs. The specific simdgroup re-org from the paper (`m/256` along M × `n₁ × n₂` along K) needs concrete numbers picked for our `gs=64` case — TBD as M0/M2 work.

5. **One alignment trap.** Pick shapes where K%512==0 and N%8==0 hold, so the baseline is `qmv_fast` rather than the slower general `qmv`. Misaligned shapes would make the comparison artificially favorable to us.
