import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load bb[1][i] values for all i
    bb_ptr_1 = bb_ptr + LEN_2D + i_offsets
    bb_1_i = tl.load(bb_ptr_1, mask=i_mask, other=0.0)
    
    # Load d[1]
    d_1 = tl.load(d_ptr + 1)
    
    # Load a[0] and compute a[1]
    a_0 = tl.load(a_ptr)
    a_1 = 1.0 - a_0
    
    # Store a[1]
    tl.store(a_ptr + 1, a_1)
    
    # Compute aa[1][i] = a[1] + bb[1][i] * d[1]
    aa_1_i = a_1 + bb_1_i * d_1
    aa_ptr_1 = aa_ptr + LEN_2D + i_offsets
    tl.store(aa_ptr_1, aa_1_i, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for j in range(1, LEN_2D):
        # Load a[j-1], compute a[j] = 1.0 - a[j-1]
        a_prev = a[j-1].item()
        a_j = 1.0 - a_prev
        a[j] = a_j
        
        # Compute aa[j][:] = a[j] + bb[j][:] * d[j]
        aa[j] = a_j + bb[j] * d[j]