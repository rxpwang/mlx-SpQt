import mlx.core as mx
import mlx_spqt

def test_smoke_threadgroup():
    '''
    verify the threadgroup primitive by launching N threads conducting simd reduction to compute the sum of 1.0 across all threads. if threadgroup is working correctly, the final result should be N.
    setup:
       single threadgroup with N threads
       each thread starts with 1.0
       two stage reduce: 
    expected: out[0] = N (the sum of 1.0 across all threads)
    '''

    N = 1024
    out = mlx_spqt.smoke_threadgroup(N=N)
    mx.eval(out)
    got = out[0].item()
    expected = float(N)
    assert abs(got - expected) < 1e-5, f"Expected {expected}, but got {got}"
    print("smoke_threadgroup: PASS")

if __name__ == "__main__":
    test_smoke_threadgroup()
