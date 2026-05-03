import mlx.core as mx
import mlx_spqt

SHAPES = [
    (1024, 1024),
    (2048, 2048),                                                                                          
    (4096, 4096),
    (8192, 8192),                                                                                          
    (4096, 11008),
    (11008, 4096),
    (4096, 16384),
]

def test_zigzag_qmv_dense(M, K):
    """M1 done criterion: dense zigzag-GEMV matches dequantize-then-fp-matmul.
                                                                                                             
      Per investigations/M1-zigzag-layout.md §7. Threshold mirrors MLX's own
      test_quantized.py::test_qmv (1e-3 in fp32).                                                            
    """    
    #M, K = 4096, 4096
    group_size, bits = 64, 4

    mx.random.seed(42)
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)
    x = mx.random.normal(shape=(1, K)).astype(mx.float16)

    # step 1: quantize w into zigzag format
    w_zz, scales, biases = mlx_spqt.quantize_zigzag(w, group_size=group_size, bits=bits)

    # step 2: our kernel
    y_zz = mlx_spqt.zigzag_qmv_dense(x, w_zz, scales, biases, group_size=group_size, bits=bits)

    # shape + dtype checks
    assert y_zz.shape == (1, M), f"Expected output shape (1, {M}), but got {y_zz.shape}"
    assert y_zz.dtype == mx.float32, f"Expected output dtype float32, but got {y_zz.dtype}"

    # step 3: reference fp matmul with dequantized weights
    sc_f32 = scales.astype(mx.float32)                                                                             
    b_f32 = biases.astype(mx.float32) 
    w_dequant = mlx_spqt.dequantize_zigzag(w_zz, sc_f32, b_f32, group_size=group_size, bits=bits)
    y_ref = x.astype(mx.float32) @ w_dequant.astype(mx.float32).T 

    # step 4: compare with a reasonable error threshold
    err = (y_zz.astype(mx.float32) - y_ref).abs().max().item()
    print(f"Max absolute error: {err}")
    assert err < 1e-3, f"Max absolute error {err} exceeds expected threshold for zigzag GEMV"

if __name__ == "__main__":
    for M, K in SHAPES:
        print(f"Testing zigzag_qmv_dense with shape M={M}, K={K}...")
        test_zigzag_qmv_dense(M, K)
    print("zigzag_qmv_dense test passed!")