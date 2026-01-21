import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block (starting from j=1)
    j_block = tl.program_id(0)
    j_offsets = j_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # For each valid j in this block, compute sequentially
    for block_idx in range(BLOCK_SIZE):
        j = j_block * BLOCK_SIZE + block_idx + 1
        if j < LEN_2D:
            # Load a[j-1] 
            a_prev = tl.load(a_ptr + j - 1)
            
            # Compute a[j] = 1.0 - a[j-1]
            a_new = 1.0 - a_prev
            
            # Store a[j]
            tl.store(a_ptr + j, a_new)
            
            # Load bb[j][i] and d[j]
            bb_val = tl.load(bb_ptr + j * LEN_2D + i)
            d_val = tl.load(d_ptr + j)
            
            # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
            aa_val = a_new + bb_val * d_val
            
            # Store aa[j][i]
            tl.store(aa_ptr + j * LEN_2D + i, aa_val)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Parallel loop over i, sequential execution over j within each i
    for i in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)  # j from 1 to LEN_2D-1
        s256_kernel[grid](a, aa, bb, d, i, LEN_2D, BLOCK_SIZE)