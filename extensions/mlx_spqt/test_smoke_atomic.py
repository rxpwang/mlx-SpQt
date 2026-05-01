import mlx.core as mx
import mlx_spqt

def test_smoke_atomic_basic():
    '''
    verify that when multiple threads update the same output location, the final result is correct (i.e. all updates are atomic and none are lost).
    setup: launch N threads that each add 1.0 to out[0]. if updates are atomic, the final value at out[0] should be N.
    '''
    N = 1024
    out = mlx_spqt.smoke_atomic(N=N)
    mx.eval(out)
    expected = mx.array([N], dtype=mx.float32)
    assert mx.all(out[0] == expected).item(), f"Expected {expected}, but got {out}"
    print("smoke_atomic_basic: PASS")

def test_smoke_atomic_stress():
    '''
    stress test the atomicity by launching a large number of threads that update the same output location.
    setup: launch 10*N threads that each add 1.0 to out[0]. if updates are atomic, the final value at out[0] should be 10*N.
    '''
    N = 1024
    num_threads = 10 * N
    out = mlx_spqt.smoke_atomic(N=num_threads)
    mx.eval(out)
    expected = mx.array([num_threads], dtype=mx.float32)
    assert mx.all(out[0] == expected).item(), f"Expected {expected}, but got {out}"
    print("smoke_atomic_stress: PASS")

if __name__ == "__main__":
    test_smoke_atomic_basic()
    test_smoke_atomic_stress()