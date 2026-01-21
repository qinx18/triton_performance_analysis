import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get i indices for this block
    i_block = tl.program_id(0)
    i_offsets = i_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] (scalar)
    a_prev = tl.load(a_ptr + j - 1)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j] (scalar)
    tl.store(a_ptr + j, a_new)
    
    # Vectorized computation for all i values in this block
    # Load bb[j][i] values
    bb_offsets = j * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    
    # Load d[j] (scalar, broadcast to vector)
    d_val = tl.load(d_ptr + j)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new + bb_vals * d_val
    
    # Store aa[j][i] values
    aa_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over j, parallel execution over i
    for j in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s256_kernel[grid](a, aa, bb, d, j, LEN_2D, BLOCK_SIZE)