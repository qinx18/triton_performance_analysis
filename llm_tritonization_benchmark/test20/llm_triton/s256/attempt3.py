import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D: tl.constexpr):
    # For j=1 to LEN_2D-1, compute sequentially
    for j in range(1, LEN_2D):
        # a[j] = 1.0 - a[j-1]
        a_prev = tl.load(a_ptr + j - 1)
        a_j = 1.0 - a_prev
        tl.store(a_ptr + j, a_j)
        
        # aa[j][i] = a[j] + bb[j][i] * d[j]
        bb_ji = tl.load(bb_ptr + j * LEN_2D + i)
        d_j = tl.load(d_ptr + j)
        aa_ji = a_j + bb_ji * d_j
        tl.store(aa_ptr + j * LEN_2D + i, aa_ji)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    
    # Launch one thread per i value
    for i in range(LEN_2D):
        s256_kernel[(1,)](
            a, aa, bb, d, i,
            LEN_2D=LEN_2D
        )