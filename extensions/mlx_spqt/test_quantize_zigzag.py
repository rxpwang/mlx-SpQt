import mlx.core as mx
import mlx_spqt

def test_round_trip():
    M, K = 4096, 4096
    group_size, bits = 64, 4

    mx.random.seed(42)
    w = mx.random.normal(shape=(M, K)).astype(mx.float16)

    w_q, scales, biases = mlx_spqt.quantize_zigzag(w, group_size=group_size, bits=bits)

    # shape contracts
    assert w_q.shape == (M // group_size, K, group_size * bits // 32), f"Unexpected w_q shape: {w_q.shape}"
    assert scales.shape == (M // group_size, K), f"Unexpected scales shape: {scales.shape}"
    assert biases.shape == (M // group_size, K), f"Unexpected biases shape: {biases.shape}"
    assert w_q.dtype == mx.uint32, f"Expected w_q dtype uint32, but got {w_q.dtype}"
    assert scales.dtype == mx.float16, f"Expected scales dtype float16, but got {scales.dtype}"
    assert biases.dtype == mx.float16, f"Expected biases dtype float16, but got {biases.dtype}"

    # reconstruction
    w_recon = mlx_spqt.dequantize_zigzag(w_q, scales, biases, group_size=group_size, bits=bits)
    assert w_recon.shape == w.shape, f"Expected recon shape {w.shape}, but got {w_recon.shape}"
    assert w_recon.dtype == w.dtype, f"Expected recon dtype {w.dtype}, but got {w_recon.dtype}"

    # error bound check
    err = (w - w_recon).abs().max().item()
    print(f"Max absolute error: {err}")
    assert err < 0.5, f"Max absolute error {err} exceeds expected bound for 4-bit quantization" 

if __name__ == "__main__":
    test_round_trip()
    print("quantize_zigzag round-trip test passed!")