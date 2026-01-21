import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D: tl.constexpr):
    for j in range(1, LEN_2D):
        # Load a[j-1]
        a_prev = tl.load(a_ptr + j - 1)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_new = 1.0 - a_prev
        
        # Store a[j]
        tl.store(a_ptr + j, a_new)
        
        # Load bb[j][i] and d[j]
        bb_offset = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_offset)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_val = a_new + bb_val * d_val
        
        # Store aa[j][i]
        aa_offset = j * LEN_2D + i
        tl.store(aa_ptr + aa_offset, aa_val)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    
    # Launch one kernel per i value
    for i in range(LEN_2D):
        grid = (1,)
        s256_kernel[grid](a, aa, bb, d, i, LEN_2D)