import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i: tl.constexpr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(0)
    
    # Calculate block offsets for j dimension (starting from 1)
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + 1
    j_mask = j_offsets < LEN_2D
    
    # Load a[j-1] values
    a_prev_offsets = j_offsets - 1
    a_prev_vals = tl.load(a_ptr + a_prev_offsets, mask=j_mask, other=0.0)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j] values
    tl.store(a_ptr + j_offsets, a_vals, mask=j_mask)
    
    # Load d[j] values
    d_vals = tl.load(d_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i] values
    aa_offsets = j_offsets * LEN_2D + i
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Loop over i dimension (outer loop)
    for i in range(LEN_2D):
        # Launch kernel for all j values (j from 1 to LEN_2D-1)
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s256_kernel[grid](a, aa, bb, d, i=i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)