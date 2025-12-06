import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # Load from read-only copy for dependency-free reads
    a_prev = tl.load(a_copy_ptr + (j_offsets - 1), mask=j_mask)
    bb_vals = tl.load(bb_ptr + j_offsets * LEN_2D + i, mask=j_mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=j_mask)
    
    # Compute new a values
    a_new = 1.0 - a_prev
    
    # Store new a values
    tl.store(a_ptr + j_offsets, a_new, mask=j_mask)
    
    # Compute and store aa values
    aa_vals = a_new + bb_vals * d_vals
    tl.store(aa_ptr + j_offsets * LEN_2D + i, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(LEN_2D):
        # Create read-only copy for this iteration
        a_copy = a.clone()
        
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s256_kernel[grid](
            a, a_copy, aa, bb, d, i, LEN_2D, BLOCK_SIZE
        )