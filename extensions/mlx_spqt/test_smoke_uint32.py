import numpy as np
import mlx.core as mx
import mlx_spqt

def test_smoke_uint32_basic():
    # pack 8 nibbles: 0,1,2,3,4,5,6,7 → into one uint32
    packed = np.array([0x76543210], dtype=np.uint32)
    w = mx.array(packed)
    out = mlx_spqt.smoke_uint32(w, N=8)
    mx.eval(out)
    #print(out)
    expected = mx.array([0,1,2,3,4,5,6,7], dtype=mx.float32)
    assert mx.all(out == expected).item(), f"Expected {expected}, but got {out.numpy()}"

def test_smoke_uint32_larger():
    N = 256
    nibbles = np.arange(N) % 16
    packed = np.zeros(N // 8, dtype=np.uint32)
    for i in range(N):
        packed[i // 8] |= (nibbles[i] << (4 * (i % 8)))
    w = mx.array(packed)
    out = mlx_spqt.smoke_uint32(w, N=N)
    mx.eval(out)
    expected = nibbles.astype(np.float32)
    got = np.array(out)
    assert np.array_equal(got, expected), f"Expected {expected}, but got {got}"

if __name__ == "__main__":
    test_smoke_uint32_basic()
    test_smoke_uint32_larger()
    print("smoke_uint32 tests passed!")