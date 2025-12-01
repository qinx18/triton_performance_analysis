import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Vectorized over j dimension
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_indices = offsets + 1  # j starts from 1
    
    mask = j_indices < LEN_2D
    
    # Current position aa[i][j]
    aa_idx = i_val * LEN_2D + j_indices
    # Dependency position aa[i-1][j-1]
    dep_idx = (i_val - 1) * LEN_2D + (j_indices - 1)
    # bb[i][j] position
    bb_idx = i_val * LEN_2D + j_indices
    
    # Load values
    aa_dep = tl.load(aa_ptr + dep_idx, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_idx, mask=mask)
    
    # Compute
    result = aa_dep + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_idx, result, mask=mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Sequential over i, parallel over j
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
        s119_kernel[grid](aa, bb, i, LEN_2D, BLOCK_SIZE)