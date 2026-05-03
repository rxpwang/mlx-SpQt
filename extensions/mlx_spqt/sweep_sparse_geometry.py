"""Sweep (num_simdgroups, threadgroups_per_band) combinations for the sparse                                   
kernel across shapes × densities. Records timing per combo, prints summary                                     
table and recommends shape-specific best combo.                                                                
                                                                                                                
Requires zigzag_qmv_sparse to accept num_simdgroups and threadgroups_per_band                                  
as runtime kwargs (refactor lands in the same commit).                                                         
                                                                                                                
Runtime: ~5-7 minutes (6 combos × 7 shapes × 4 densities).                                                     
"""                                                                                                            
                
import os                                                                                                      
import sys      
import time
from collections import defaultdict
                                                                                                                
import mlx.core as mx
import mlx_spqt                                                                                                
import numpy as np

_MLX_BENCHMARKS = os.path.join(                                                                                
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "python"
)                                                                                                              
sys.path.insert(0, _MLX_BENCHMARKS)
                                                                                                                
                                                                                                                
# ── Configuration ──────────────────────────────────────────────────────────
SHAPES = [                                                                                                     
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),                                                                                              
    (4096, 11008),
    (11008, 4096),                                                                                             
    (4096, 16384),
]                                                                                                              
DENSITIES = [0.5, 0.25]   # focus on the densities where sparse matters
                                                                                                                
# (num_simdgroups, threadgroups_per_band) — must have a [[host_name]]                                          
# instantiation in .metal for each combo.                                                                      
COMBOS = [(2, 2), (2, 4), (4, 1), (4, 2), (4, 4), (4, 8), (8, 2), (1, 4), (2, 1), (2, 8), (8, 1)]                                                      
                                                                                                                
GS, BITS = 64, 4                                                                                               
LOOPS = 32                                                                                                     
                                                                                                                
                                                                                                                
def make_sparse_idx(K, density):                                                                               
    n = max(16, int(K * density))                                                                              
    n = (n // 16) * 16
    rng = np.random.default_rng(seed=K * 1000 + n)                                                             
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)                                         
    idx_np = np.insert(idx_np, 0, n)                                                                           
    return mx.array(idx_np)                                                                                    
                                                                                                                
                
def time_msec(fn, num_iters=50, num_warmup=5):                                                                 
    for _ in range(num_warmup):
        mx.eval(fn())                                                                                          
    tic = time.perf_counter()
    for _ in range(num_iters):                                                                                 
        mx.eval(fn())
    toc = time.perf_counter()                                                                                  
    return 1e3 * (toc - tic) / num_iters
                                                                                                                
                                                                                                                
def bench_shape_combo(M, K, w_zz, sc_zz, b_zz, x, idx, density, nsg, tg):
    """Time the sparse kernel at a specific (NSG, TG) combo."""                                                
    def run():  
        y = mlx_spqt.zigzag_qmv_sparse(                                                                        
            x, idx, w_zz, sc_zz, b_zz,
            group_size=GS, bits=BITS,                                                                          
            num_simdgroups=nsg, threadgroups_per_band=tg,
        )                                                                                                      
        for _ in range(LOOPS - 1):
            y = y + mlx_spqt.zigzag_qmv_sparse(                                                                
                x, idx, w_zz, sc_zz, b_zz,
                group_size=GS, bits=BITS,                                                                      
                num_simdgroups=nsg, threadgroups_per_band=tg,
            )                                                                                                  
        return y                                                                                               
    return time_msec(run) * 1000.0 / LOOPS  # μs / kernel call
                                                                                                                
                                                                                                                
def correctness_check(M, K, w_zz, sc_zz, b_zz, x, idx, nsg, tg):                                               
    """Quick sanity: kernel output matches fp32 reference within 1e-3."""                                      
    y = mlx_spqt.zigzag_qmv_sparse(                                                                            
        x, idx, w_zz, sc_zz, b_zz,
        group_size=GS, bits=BITS,                                                                              
        num_simdgroups=nsg, threadgroups_per_band=tg,
    )                                                                                                          
    sc_f32 = sc_zz.astype(mx.float32)                                                                          
    b_f32  = b_zz.astype(mx.float32)
    w_dq_f32 = mlx_spqt.dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=GS, bits=BITS)                       
    idx_k = idx[1:]                                                                                            
    y_ref = x.astype(mx.float32)[:, idx_k] @ w_dq_f32[:, idx_k].T                                              
    err = (y - y_ref).abs().max().item()                                                                       
    return err  
def main():                                                                                                    
    # results[(M, K, density)][(nsg, tg)] = us_per_call
    results = defaultdict(dict)                                                                                
    correctness = defaultdict(dict)
                                                                                                                
    print(f"Sweeping {len(COMBOS)} combos × {len(SHAPES)} shapes × {len(DENSITIES)} densities")                
    print(f"Combos: {COMBOS}")                                                                                 
    print()                                                                                                    
                
    for (M, K) in SHAPES:                                                                                      
        # Setup once per shape
        mx.random.seed(0)                                                                                      
        w = mx.random.normal(shape=(M, K)).astype(mx.float16)
        x = mx.random.normal(shape=(1, K)).astype(mx.float16)                                                  
        w_zz, sc_zz, b_zz = mlx_spqt.quantize_zigzag(w, group_size=GS, bits=BITS)                              
        mx.eval(w_zz, sc_zz, b_zz, x)                                                                          
                                                                                                                
        idx_per_density = {d: make_sparse_idx(K, d) for d in DENSITIES}                                        
        mx.eval(*idx_per_density.values())
                                                                                                                
        for density in DENSITIES:
            idx = idx_per_density[density]                                                                     
                
            for (nsg, tg) in COMBOS:                                                                           
                # Correctness
                err = correctness_check(M, K, w_zz, sc_zz, b_zz, x, idx, nsg, tg)                              
                correctness[(M, K, density)][(nsg, tg)] = err                                                  

                if err > 1e-3:                                                                                 
                    print(f"  WARN: M={M} K={K} d={density} (NSG={nsg}, TG={tg}) "
                        f"err={err:.2e} > 1e-3")                                                             
                    results[(M, K, density)][(nsg, tg)] = float('inf')
                    continue                                                                                   
                                                                                                                
                # Bench
                us = bench_shape_combo(M, K, w_zz, sc_zz, b_zz, x, idx, density, nsg, tg)                      
                results[(M, K, density)][(nsg, tg)] = us
                                                                                                                
            print(f"  M={M:>5}, K={K:>5}, density={density}: done")                                            
                                                                                                                
    # ── Summary table ──────────────────────────────────────────────────────                                  
    print()     
    print("=" * 78)                                                                                            
    print(f"  Per-shape × density: best combo (μs/call)")
    print("=" * 78)                                                                                            
    print()
                                                                                                                
    header_cols = "  ".join(f"({n},{t})" for (n, t) in COMBOS)                                                 
    print(f"  {'shape':<22} {'density':>8}  {header_cols:<55} {'best':>10}")                                   
    print(f"  {'-' * 22} {'-' * 8}  {'-' * 55} {'-' * 10}")                                                    
                                                                                                                
    best_for_shape_density = {}                                                                                
    for (M, K) in SHAPES:                                                                                      
        for density in DENSITIES:                                                                              
            row = results[(M, K, density)]
            cells = [f"{row[(n, t)]:.0f}" for (n, t) in COMBOS]
            best = min(row.items(), key=lambda kv: kv[1])                                                      
            best_for_shape_density[(M, K, density)] = best                                                     
            cells_str = "  ".join(f"{c:>5}" for c in cells)                                                    
            print(f"  ({M:>5}, {K:>5})       {density:>6.2f}    {cells_str}   "                                
                f"{best[0]} {best[1]:.0f}")                                                                  
                                                                                                                
    # ── Recommendation ─────────────────────────────────────────────────────                                  
    print()                                                                                                    
    print("=" * 78)                                                                                            
    print(f"  Best combo per shape (averaged across densities):")
    print("=" * 78)                                                                                            
    print()     
    print(f"  {'shape':<22} {'best_combo':<12} {'us @ d=0.5':>12} {'us @ d=0.25':>14}")                        
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 14}")                                                    
    for (M, K) in SHAPES:                                                                                      
        # rank combos by sum across densities                                                                  
        scores = {combo: sum(results[(M, K, d)][combo] for d in DENSITIES) for combo in COMBOS}                
        best_combo = min(scores.items(), key=lambda kv: kv[1])[0]                                              
        print(f"  ({M:>5}, {K:>5})       {str(best_combo):<12} "                                               
            f"{results[(M, K, 0.5)][best_combo]:>10.0f} us "                                                 
            f"{results[(M, K, 0.25)][best_combo]:>12.0f} us")                                                
                                                                                                                
                                                                                                                
if __name__ == "__main__":                                                                                     
    main()      
