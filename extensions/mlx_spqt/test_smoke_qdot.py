import numpy as np
import mlx.core as mx
import mlx_spqt

def test_smoke_qdot():
    """Verify qdot<float, 16, 4> from quantized.h is callable and correct.
                                                                                                             
    Setup:
    - 16 nibbles 0..15 packed across two uint32s                                                         
    - x = ones(16)                                                                                       
    - scale = 1.0, bias = 0.0 (hardcoded in the kernel)                                                  
    Expected: sum_{i=0..15}(1.0 * i) = 120.0 at out[0]                                                     
    """        
    packed = np.array([0x76543210, 0xFEDCBA98], dtype=np.uint32)
    w = mx.array(packed)
    x = mx.ones(16, dtype=mx.float32)

    out = mlx_spqt.smoke_qdot(w, x, N=16)
    mx.eval(out)

    arr = np.array(out)
    print(f"out[0]  = {arr[0]}")
    print(f"expected: 120.0")                                                                              
    print(f"out[1:]  = {arr[1:]}  (uninitialized; ignore)")
                                                                                                            
    assert abs(arr[0] - 120.0) < 1e-5, f"out[0] = {arr[0]}, expected 120.0"                                
    print("smoke_qdot: PASS")                                                                              
                                                                                                             
                  
if __name__ == "__main__":
    test_smoke_qdot()