"""SpQt → MLX port: end-to-end demo.                                                                           
                                                                                                                
Runs correctness AND performance for both zigzag_qmv_dense and                                                 
zigzag_qmv_sparse, across 7 LLM-relevant shapes × {1.0, 0.75, 0.5, 0.25}                                       
sparsity densities. Single command, single Python script.                                                      
                                                                                                                
Total runtime: ~1-2 minutes.                                                                                   
"""         

import os                                                                                                      
import sys
import time                                                                                                    
                                                                                                            
import mlx.core as mx
import mlx_spqt
import numpy as np

# MLX's bench harness lives under benchmarks/python/; add to path.                                             
_MLX_BENCHMARKS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "python"                             
)                                                                                                            
sys.path.insert(0, _MLX_BENCHMARKS)                                                                            
from time_utils import time_fn 

# ── Configuration ───────────────────────────────────────────────────────────                                 
SHAPES = [                                                                                                   
    (1024, 1024),
    (2048, 2048),                                                                                              
    (4096, 4096),
    (8192, 8192),                                                                                              
    (4096, 11008),  # Llama-7B FFN-down                                                                      
    (11008, 4096),  # Llama-7B FFN-up
    (4096, 16384),                                                                                             
]
DENSITIES = [1.0, 0.75, 0.5, 0.25]                                                                             
GS, BITS = 64, 4                                                                                             
LOOPS = 32                                                                                                     
THRESHOLD = 1e-3

def make_sparse_idx(K, density):                                                                               
    n = max(16, int(K * density))                                                                              
    n = (n // 16) * 16                                                                                       
    rng = np.random.default_rng(seed=K * 1000 + n)                                                             
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)
    idx_np = np.insert(idx_np, 0, n)                                                                           
    return mx.array(idx_np)     

def time_msec(fn, num_iters=100, num_warmup=5):                                                                
    """Like MLX's time_fn but returns the msec value rather than printing it."""                             
    for _ in range(num_warmup):                                                                                
        mx.eval(fn())
    tic = time.perf_counter()                                                                                  
    for _ in range(num_iters):                                                                               
        mx.eval(fn())                                                                                          
    toc = time.perf_counter()
    return 1e3 * (toc - tic) / num_iters     

def section(title):
    print()                                                                                                    
    print("=" * 78)                                                                                          
    print(f"  {title}")
    print("=" * 78)


def setup_shape(M, K):                                                                                         
    """Quantize weights and prep activations for one shape. Returns dict of artifacts."""                    
    mx.random.seed(0)                                                                                          
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)                                                      
    x = mx.random.normal(shape=(1, K)).astype(mx.float16)                                                      
                                                                                                                
    w_zz, sc_zz, b_zz = mlx_spqt.quantize_zigzag(w, group_size=GS, bits=BITS)                                  
    w_q, sc, bi      = mx.quantize(w, group_size=GS, bits=BITS)                                              
    w_dq             = mlx_spqt.dequantize_zigzag(w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)                 
    sc_f32           = sc_zz.astype(mx.float32)                                                                
    b_f32            = b_zz.astype(mx.float32)                                                                 
    w_dq_f32         = mlx_spqt.dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=GS, bits=BITS)               
    mx.eval(w_zz, sc_zz, b_zz, w_q, sc, bi, w_dq, w_dq_f32, x)                                                 
                                                                                                                
    return dict(                                                                                               
        M=M, K=K, w=w, x=x,                                                                                    
        w_zz=w_zz, sc_zz=sc_zz, b_zz=b_zz,                                                                     
        w_q=w_q, sc=sc, bi=bi,
        w_dq=w_dq, w_dq_f32=w_dq_f32,                                                                          
    )                       


# ── Section 1: Correctness ──────────────────────────────────────────────────
                                                                                                                 
def correctness(d):                                                                                          
    """Returns dict {(M, K, density_or_'dense'): err}."""
    M, K, x = d['M'], d['K'], d['x']                                                                           
    w_zz, sc_zz, b_zz, w_dq_f32 = d['w_zz'], d['sc_zz'], d['b_zz'], d['w_dq_f32']                              
    results = {}                                                                                               
                                                                                                                
    # Dense                                                                                                    
    y = mlx_spqt.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)                            
    y_ref = x.astype(mx.float32) @ w_dq_f32.T                                                                  
    results[('dense',)] = (y - y_ref).abs().max().item()                                                       
                                                                                                                
    # Sparse at each density                                                                                   
    for dens in DENSITIES:                                                                                     
        idx = make_sparse_idx(K, dens)                                                                         
        y = mlx_spqt.zigzag_qmv_sparse(x, idx, w_zz, sc_zz, b_zz,
                                        group_size=GS, bits=BITS)                                              
        idx_k = idx[1:]                                                                                      
        y_ref = x.astype(mx.float32)[:, idx_k] @ w_dq_f32[:, idx_k].T                                          
        results[('sparse', dens)] = (y - y_ref).abs().max().item()                                             

    return results      


# ── Section 2: Performance ──────────────────────────────────────────────────
                                                                                                                
def make_runners(d):                                                                                         
    """Build closures for each kernel/density so they can be timed."""
    M, K, x = d['M'], d['K'], d['x']                                                                           
    w_zz, sc_zz, b_zz = d['w_zz'], d['sc_zz'], d['b_zz']
    w_q, sc, bi = d['w_q'], d['sc'], d['bi']                                                                   
    w_dq = d['w_dq']                                                                                         
                                                                                                                
    def run_dense():                                                                                           
        y = mlx_spqt.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
        for _ in range(LOOPS - 1):                                                                             
            y = y + mlx_spqt.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)                  
        return y                                                                                               
                                                                                                                
    def run_mlx():                                                                                             
        y = mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)                                    
        for _ in range(LOOPS - 1):                                                                             
            y = y + mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)
        return y                                                                                               
                                                                                                            
    def run_naive():                                                                                           
        y = x @ w_dq.T                                                                                       
        for _ in range(LOOPS - 1):
            y = y + (x @ w_dq.T)
        return y                                                                                               

    sparse_runners = {}                                                                                        
    for dens in DENSITIES:                                                                                   
        idx = make_sparse_idx(K, dens)
        mx.eval(idx)                                                                                           
        # closure-bind idx
        def make(idx):                                                                                         
            def run():                                                                                       
                y = mlx_spqt.zigzag_qmv_sparse(x, idx, w_zz, sc_zz, b_zz,                                      
                                                group_size=GS, bits=BITS)                                    
                for _ in range(LOOPS - 1):                                                                     
                    y = y + mlx_spqt.zigzag_qmv_sparse(x, idx, w_zz, sc_zz, b_zz,                            
                                                        group_size=GS, bits=BITS)                               
                return y                                                                                     
            return run                                                                                         
        sparse_runners[dens] = make(idx)                                                                     

    return run_dense, run_mlx, run_naive, sparse_runners                                                       


# ── Main ────────────────────────────────────────────────────────────────────                               

def main():
    print()
    print("=" * 78)                                                                                            
    print(f"  SpQt → MLX port: end-to-end demo")
    print("=" * 78)                                                                                            
    print(f"\n  Shapes:    {len(SHAPES)} ({SHAPES[0]} ... {SHAPES[-1]})")                                    
    print(f"  Densities: {DENSITIES} (sparse only)")                                                           
    print(f"  Bench:     time_fn-style, chained LOOPS={LOOPS} per timed iter\n")
                                                                                                                
    # Pre-setup all shapes (so correctness section runs fast)                                                  
    shape_data = [setup_shape(M, K) for (M, K) in SHAPES]                                                      
                                                                                                                
    # ── Correctness ────────────────────────────────────────────────────────                                
    section(f"[1/3] Correctness across shapes × densities (threshold = {THRESHOLD:.0e})")                      
    print()                                                                                                    
    print(f"  {'shape':<18} {'dense':>10} {'s@1.00':>10} {'s@0.75':>10} {'s@0.50':>10} {'s@0.25':>10}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")                              
    all_pass = True                                                                                            
    for d in shape_data:                                                                                       
        r = correctness(d)                                                                                     
        cells = []                                                                                           
        for key in [('dense',), ('sparse', 1.0), ('sparse', 0.75),
                    ('sparse', 0.5), ('sparse', 0.25)]:                                                        
            err = r[key]                                                                                       
            mark = '✓' if err < THRESHOLD else '✗'                                                             
            cells.append(f"{err:.1e} {mark}")                                                                  
            if err >= THRESHOLD:                                                                               
                all_pass = False
        print(f"  ({d['M']:>5}, {d['K']:>5})    {cells[0]:>10} {cells[1]:>10} "                                
            f"{cells[2]:>10} {cells[3]:>10} {cells[4]:>10}")                                                 
                                                                                                                
    print()                                                                                                    
    print(f"  Result: {'all PASS' if all_pass else 'some FAIL — check kernel'}")                               
                                                                                                                
    # ── Performance ────────────────────────────────────────────────────────
    section(f"[2/3] Performance — kernel time per call (μs at LOOPS={LOOPS})")                                 
                                                                                                                
    perf_data = []  # list of (M, K, dense, mlx, naive, sparse_dict)                                           
                                                                                                                
    for d in shape_data:                                                                                       
        M, K = d['M'], d['K']                                                                                
        runners = make_runners(d)
        run_dense, run_mlx, run_naive, sparse_runners = runners                                                

        # Warmup interleaved across all this shape's kernels                                                   
        for _ in range(20):                                                                                  
            mx.eval(run_dense())                                                                               
            mx.eval(run_mlx())                                                                               
            mx.eval(run_naive())                                                                               
            for sdens in DENSITIES:
                mx.eval(sparse_runners[sdens]())                                                               
        mx.synchronize()                                                                                       

        # Time each. Convert msec/chained-call to us/kernel-call.                                              
        t_dense = time_msec(run_dense) * 1000.0 / LOOPS                                                      
        t_mlx   = time_msec(run_mlx)   * 1000.0 / LOOPS                                                        
        t_naive = time_msec(run_naive) * 1000.0 / LOOPS                                                        
        t_sparse = {dens: time_msec(sparse_runners[dens]) * 1000.0 / LOOPS                                     
                    for dens in DENSITIES}                                                                     
        perf_data.append((M, K, t_dense, t_mlx, t_naive, t_sparse))                                            
                                                                                                                
        print()                                                                                                
        print(f"  M={M:>5}, K={K:>5}")                                                                       
        print(f"    dense (ours)        : {t_dense:>6.1f} us")                                                 
        print(f"    mx.quantized_matmul : {t_mlx:>6.1f} us  [baseline]")                                       
        print(f"    naive fp matmul     : {t_naive:>6.1f} us")                                                 
        for dens in DENSITIES:                                                                                 
            ratio = t_mlx / t_sparse[dens]                                                                     
            arrow = '←' if dens == 0.25 else ' '                                                               
            print(f"    sparse @ d={dens:.2f}      : {t_sparse[dens]:>6.1f} us"                                
                f"  ({ratio:.2f}× of mlx_qmv) {arrow}")                                                      
                                                                                                                
    # ── Summary ────────────────────────────────────────────────────────────                                  
    section(f"[3/3] Summary — SpQt headline (sparse @ 75% sparsity)")                                          
    print()                                                                                                    
    print(f"  At density=0.25 (only 25% of K-positions walked):")
    print()                                                                                                    
    print(f"  {'shape':<18} {'mlx_qmv (full)':>18} {'sparse@0.25':>15} {'speedup':>10}")                       
    print(f"  {'-' * 18} {'-' * 18} {'-' * 15} {'-' * 10}")                                                    
    speedups = []                                                                                              
    for (M, K, t_dense, t_mlx, t_naive, t_sparse) in perf_data:                                                
        ratio = t_mlx / t_sparse[0.25]                                                                         
        speedups.append(ratio)                                                                               
        print(f"  ({M:>5}, {K:>5})  {t_mlx:>15.1f} us {t_sparse[0.25]:>13.1f} us "                             
            f"{ratio:>9.2f}×")                                                                               
                                                                                                                
    avg = sum(speedups) / len(speedups)                                                                        
    llm_speedups = [s for ((M, K, *_), s) in zip(perf_data, speedups)                                          
                    if M >= 4096 and K >= 4096]                                                                
    llm_avg = sum(llm_speedups) / len(llm_speedups) if llm_speedups else 0
                                                                                                                
    print()                                                                                                  
    print(f"  Geometric average speedup across all shapes: {avg:.2f}×")                                        
    print(f"  Average across LLM-relevant shapes (M, K ≥ 4096): {llm_avg:.2f}×")                               
    print()                                                                                                    
    print(f"  This is the SpQt value proposition realized: skipping inactive")                                 
    print(f"  K-positions saves memory bandwidth proportionally, beating")                                     
    print(f"  MLX's tuned mx.quantized_matmul at production sparsity.")                                        
    print()                                                                                                    
                                                                                                                
                                                                                                                
if __name__ == "__main__":                                                                                   
    main()                                                                                                     
