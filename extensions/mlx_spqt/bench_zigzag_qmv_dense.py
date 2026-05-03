"""
Throughput bench for zigzag_qmv_dense vs MLX baselines.
                                                                                                            
Uses MLX's canonical bench harness (benchmarks/python/time_utils.py:time_fn):
warmup 5 iters, time 100 iters, eval-per-iter. The chained-call pattern                                    
inside each `run_*` function (loops = LOOPS) amortizes eval-and-sync                                       
overhead and prevents MLX's graph dedup from collapsing identical calls.                                   
                                                                                                            
Reported `time_fn` msec is per chained-fn call (= LOOPS kernel dispatches).                                
Divide by LOOPS for per-kernel time.                                                                       
"""             
                                                                                                             
import os                                                                                                  
import sys   

# MLX's time_fn lives outside the extensions dir; add it to the path.                                      
_MLX_BENCHMARKS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "python"                         
)                                                                                                          
sys.path.insert(0, _MLX_BENCHMARKS)                                                                        
                                                                                                            
import mlx.core as mx                                                             
import mlx_spqt                              
from time_utils import time_fn              

LOOPS = 32       # chained kernel calls per timed iter (amortizes eval overhead)                           
#M, K = 4096, 4096                                                                                          
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

def bench_shape(M, K):
    mx.random.seed(42)
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
    x = mx.random.normal(shape=(1, K)).astype(mx.float16)

    # Pre-quantize / dequantize so the bench measures only the kernel work                                 
    w_zz, sc_zz, b_zz = mlx_spqt.quantize_zigzag(w, group_size=GS, bits=BITS)                              
    w_q, sc, bi      = mx.quantize(w, group_size=GS, bits=BITS)                                            
    w_dq             = mlx_spqt.dequantize_zigzag(w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)             
    mx.eval(w_zz, sc_zz, b_zz, w_q, sc, bi, w_dq, x)   

    # ---- Each run_* chains LOOPS dispatches via dependency on `y` ----      
    
    def run_zigzag(x):
        y = mlx_spqt.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
        for _ in range(LOOPS - 1):
            y = y + mlx_spqt.zigzag_qmv_dense(x, w_zz, sc_zz, b_zz, group_size=GS, bits=BITS)
        return y
    
    def run_mlx_qmv(x):
        y = mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)
        for _ in range(LOOPS - 1):
            y = y + mx.quantized_matmul(x, w_q, sc, bi, group_size=GS, bits=BITS)
        return y
    
    def run_fp_matmul(x):
        y = x @ w_dq.T
        for _ in range(LOOPS - 1):
            y = y + x @ w_dq.T
        return y
    
    print(f"shape: M={M}, K={K}")
    #print("(reported time is per chained-fn call, divide by LOOPS for per-kernel time)")

    for _ in range(50):                                                                                      
        mx.eval(run_zigzag(x))                                                                                 
        mx.eval(run_mlx_qmv(x))                                                                                
        mx.eval(run_fp_matmul(x))
    mx.synchronize()      

    time_fn(run_zigzag, x, msg="zigzag_qmv_dense (ours)")
    time_fn(run_mlx_qmv, x, msg="mx.quantized_matmul (baseline)")
    time_fn(run_fp_matmul, x, msg="x @ w_dq.T (fp16)")

def main():
    print(f"group_size={GS}, bits={BITS}, loops={LOOPS}")
    for M, K in SHAPES:
        bench_shape(M, K)

if __name__ == "__main__":
    main()