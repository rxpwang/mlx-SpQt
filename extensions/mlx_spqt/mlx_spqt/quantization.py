import mlx.core as mx

def quantize_zigzag(w, group_size=64, bits=4):
    """Affine-zigzag quantization. Mirrors `mx.quantize(w, mode='affine')` style.
                                                                                                             
    Rearranges w in fp space so that `group_size` output rows × 1 input column
    become contiguous along the last axis, then applies standard affine                                    
    quantization. The result is bit-equivalent to what would be produced by a                              
    future `mx.quantize(w, group_size, bits, mode='affine_zigzag')`.                                       
                                                                                                            
    Args:                                                                                                  
        w: shape (M, K). M and K must both be divisible by group_size                                      
            (well, K-divisibility comes from K being arbitrary, only M is constrained).
        group_size: quantization group size = row-band size (locked equal in zigzag).                      
        bits: bit-width. MVP supports 4 only.                                                              
                                                                                                            
    Returns:                                                                                               
        w_q:    (M/group_size, K, group_size * bits / 32)  uint32                                          
        scales: (M/group_size, K)                          w.dtype
        biases: (M/group_size, K)                          w.dtype                                         
    """
    
    M, K = w.shape
    assert M % group_size == 0, f"M={M} must be divisible by group_size-{group_size}"

    # fp-space rearrangement: (M, K) → (M/group_size, K, group_size)
    w_rearr = w.reshape(M // group_size, group_size, K).transpose(0, 2, 1) # (M/group_size, K, group_size)

    # standard mx.quantize along last axis with size group_size
    # one group per (band, k_pos) slice makes scales and biases naturally aligned with the zigzag pattern, we squeeze out the redundant last axis
    w_q, scales, biases = mx.quantize(w_rearr, group_size, bits=bits, mode='affine')
    return w_q, scales.squeeze(-1), biases.squeeze(-1)

def dequantize_zigzag(w_q, scales, biases, group_size=64, bits=4):
    """ 
    Inverse of quantize_zigzag. Mirrors `mx.dequantize(w_q, scales, biases, mode='affine')` style.
    returns: w: shape (M, K) where M = w_q.shape[0] * group_size, K = w_q.shape[1]
    """
    # re-add the trailing-1 group dim that mx.dequantize expects
    scales_3d = scales[..., None] # (M/group_size, K, 1)
    biases_3d = biases[..., None] # (M/group_size, K, 1)

    # standard dequantize returns (M/group_size, K, group_size)
    w_rearr = mx.dequantize(w_q, scales_3d, biases_3d, group_size=group_size, bits=bits, mode='affine')

    # inverse rearrangement: (M/group_size, K, group_size) → (M/group_size, group_size, K) → (M, K)
    n_bands, K, _ = w_rearr.shape
    return w_rearr.transpose(0, 2, 1).reshape(n_bands * group_size, K)