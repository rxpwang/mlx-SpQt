"""Throughput bench for zigzag_qmv_sparse vs M1 baselines.
                                                                                                                 
Sweeps (M, K) shapes × sparsity densities. At each shape:                                                      
- Dense baselines (zigzag_qmv_dense, mx.quantized_matmul) measured once.                                       
- Sparse kernel measured at each density.                                                                      
                                                                                                                
Expectation: sparse time scales ~linearly with density (memory-bound kernel,
fewer K-positions => less weight bandwidth). At density=1.0, sparse should                                     
roughly match dense; at density=0.5, sparse should be ~half of dense.                                          
                                                                                                                
Uses MLX's canonical bench harness (benchmarks/python/time_utils.py:time_fn)                                   
with chained LOOPS=32 calls per timed iter to amortize eval overhead.                                          
"""         

import os                                                                                                      
import sys      

# time_utils is co-located in benchmarks/python/                                                                    
                                                                                                                
import mlx.core as mx                        
from mlx.spqt import quantize_zigzag, dequantize_zigzag
import numpy as np                                                                              
from time_utils import time_fn              

LOOPS = 32        # chained kernel calls per timed iter
GS, BITS = 64, 4                                                                                               
                
SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),                                                                                              
    (4096, 11008),
    (11008, 4096),                                                                                             
    (4096, 16384),
]

DENSITIES = [1.0, 0.75, 0.5, 0.25]  

def make_sparse_idx(K, density):
    """Random sorted subset of K-positions, shape (n+1,) int32 with count prefix.

    Rounds n down to a multiple of 16 to satisfy the coarse-tail-free
    requirement of the current sparse kernel (full tail handling deferred).                                    
    """                                                                                                        
    n = max(16, int(K * density))                                                                              
    n = (n // 16) * 16            # coarse alignment for current kernel                                        
    rng = np.random.default_rng(seed=K * 1000 + n)                                                             
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)                                         
    idx_np = np.insert(idx_np, 0, n)                                                                           
    return mx.array(idx_np)          

def bench_shape(M, K):
    print(f"\n=== M={M}, K={K} ({M*K*BITS/8/1e6:.1f} MB weights) ===")
                                                                                                                
    mx.random.seed(42)
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)                                                      
    x = mx.random.normal(shape=(1, K)).astype(mx.float16)
                                                                                                                
    w_zz, sc_zz, b_zz = quantize_zigzag(w, group_size=GS, bits=BITS)                                  
    w_q,  sc,  bi    = mx.quantize(w, group_size=GS, bits=BITS)                                                
    w_dq             = dequantize_zigzag(w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)                 
    mx.eval(w_zz, sc_zz, b_zz, w_q, sc, bi, w_dq, x)

    # Pre-compute idx arrays for each density so bench loop measures only kernel work                          
    idx_per_density = {d: make_sparse_idx(K, d) for d in DENSITIES}                                            
    mx.eval(*idx_per_density.values())        

    # ---- Baselines (no density dependence) ----
    def run_dense(x):                                                                                          
        y = mx.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
        for _ in range(LOOPS - 1):                                                                             
            y = y + mx.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)                  
        return y                                                                                               
                                                                                                                
    def run_mlx(x):                                                                                            
        y = mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)
        for _ in range(LOOPS - 1):                                                                             
            y = y + mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)                              
        return y
                                                                                                                
    # ---- Sparse runner factory (closure-captures the density's idx) ----                                     
    def make_run_sparse(idx):
        def run(x):                                                                                            
            y = mx.zigzag_qmv_sparse(x, idx, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
            for _ in range(LOOPS - 1):                                                                         
                y = y + mx.zigzag_qmv_sparse(x, idx, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
            return y                                                                                           
        return run
    
    sparse_runners = {d: make_run_sparse(idx_per_density[d]) for d in DENSITIES}    

    # ---- Warmup (interleaved across every kernel we'll time) ----                                            
    for _ in range(30):
        mx.eval(run_dense(x))                                                                                  
        mx.eval(run_mlx(x))
        for d in DENSITIES:
            mx.eval(sparse_runners[d](x))                                                                      
    mx.synchronize()
                                                                                                                
    # ---- Time baselines ----
    time_fn(run_dense, x, msg="zigzag_qmv_dense       (full)        ")
    time_fn(run_mlx,   x, msg="mx.quantized_matmul    (full)        ")                                         

    # ---- Time sparse at each density ----                                                                    
    for d in DENSITIES:
        n = int(idx_per_density[d][0].item())                                                                  
        time_fn(sparse_runners[d], x,                                                                          
                msg=f"zigzag_qmv_sparse      density={d:.2f}  n={n:>5}")
                                                                                                                
                
def main():                                                                                                    
    print(f"group_size={GS}, bits={BITS}, loops={LOOPS}")
    print("(reported time is per chained-fn call, divide by LOOPS for per-kernel)")
    for M, K in SHAPES:
        bench_shape(M, K)


if __name__ == "__main__":
    main()