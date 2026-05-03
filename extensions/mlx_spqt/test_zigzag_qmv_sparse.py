import mlx.core as mx
import mlx_spqt
import numpy as np

SHAPES = [
    (1024, 1024),
    (2048, 2048),                                                                                          
    (4096, 4096),
    (8192, 8192),                                                                                          
    (4096, 11008),
    (11008, 4096),
    (4096, 16384),
]

# Densities to test — 1.0 is the scaffold gate (should pass with current                                       
# dense-walk stub); <1.0 will only pass after idx-driven K-walk is implemented.
DENSITIES = [1.0, 0.5]

# sparse_index format: int32 array of shape (B, K+1). +1 is for count at the beginning.
def make_sparse_idx(K, density):
    """Random sorted subset of K-positions, shape (idx_count,) int32."""
    n = max(1, int(K * density))                                                                               
    rng = np.random.default_rng(seed=K * 1000 + n)
    idx_np = np.sort(rng.choice(K, n, replace=False)).astype(np.int32)  
    idx_np = np.insert(idx_np, 0, n)  # Insert count at the beginning
    return mx.array(idx_np)

def test_shape_density(M, K, density, group_size=64, bits=4, threshold=1e-3):
    """zigzag_qmv_sparse should match `x[:, idx] @ w_dq[:, idx].T` (fp32)."""
    mx.random.seed(42)                                                                                         
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
    x = mx.random.normal(shape=(1, K)).astype(mx.float16)   

    w_zz, scales, biases = mlx_spqt.quantize_zigzag(w, group_size=group_size, bits=bits)
    idx = make_sparse_idx(K, density) 

    # Our sparse kernel
    y_zz = mlx_spqt.zigzag_qmv_sparse(                                                                         
        x, idx, w_zz, scales, biases,
        group_size=group_size, bits=bits,                                                                      
    )

    assert y_zz.shape == (1, M), f"unexpected shape: {y_zz.shape}"                                             
    assert y_zz.dtype == mx.float32

    # fp32 reference: dequantize, then matmul on the sparse subset of K                                        
    sc_f32 = scales.astype(mx.float32)
    b_f32  = biases.astype(mx.float32)                                                                         
    w_dq   = mlx_spqt.dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=group_size, bits=bits)

    idx_k = idx[1:]  
    y_ref  = x.astype(mx.float32)[:, idx_k] @ w_dq.astype(mx.float32)[:, idx_k].T
    err = (y_zz - y_ref).abs().max().item()

    pass_str = "PASS" if err < threshold else "FAIL"                                                           
    print(f"  M={M:>5}, K={K:>5}, density={density:.2f}, idx_count={idx.shape[0]:>5}: "                        
        f"err={err:.6f}  {pass_str}")                                                                        
    assert err < threshold, f"M={M} K={K} density={density}: err {err:.6f} exceeds threshold {threshold}"

def main():                                                                                                    
    print("zigzag_qmv_sparse correctness across shapes and sparsity densities")
    print("=" * 72)                                                                                            
    for M, K in SHAPES:                                                                                        
        for d in DENSITIES:                                                                                    
            test_shape_density(M, K, d)                                                                        
    print("\nAll passed.")

if __name__ == "__main__":
    main()      