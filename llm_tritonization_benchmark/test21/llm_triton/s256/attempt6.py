import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, LEN_2D: tl.constexpr, i_val: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE) + 1
    mask = j_offsets < LEN_2D
    
    # Load a[j-1] values
    a_prev = tl.load(a_ptr + j_offsets - 1, mask=mask)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_new, mask=mask)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * LEN_2D + i_val
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(LEN_2D):
        num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
        grid = (num_blocks,)
        s256_kernel[grid](a, aa, bb, d, LEN_2D, i, BLOCK_SIZE)