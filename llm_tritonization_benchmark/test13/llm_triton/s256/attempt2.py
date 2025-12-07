import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, i_val, j_val, LEN_2D: tl.constexpr):
    # Compute a[j] = 1.0 - a[j-1]
    a_prev = tl.load(a_copy_ptr + j_val - 1)
    a_val = 1.0 - a_prev
    tl.store(a_ptr + j_val, a_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    bb_offset = j_val * LEN_2D + i_val
    bb_val = tl.load(bb_ptr + bb_offset)
    d_val = tl.load(d_ptr + j_val)
    
    aa_val = a_val + bb_val * d_val
    aa_offset = j_val * LEN_2D + i_val
    tl.store(aa_ptr + aa_offset, aa_val)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    
    # Sequential j iterations
    for j in range(1, LEN_2D):
        # Create read-only copy of array a
        a_copy = a.clone()
        
        # Parallel i iterations
        for i in range(LEN_2D):
            grid = (1,)
            s256_kernel[grid](
                a, a_copy, aa, bb, d,
                i, j, LEN_2D
            )