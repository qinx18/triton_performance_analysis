import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load aa[i_val, j] for all j in this block
    aa_offsets = i_val * LEN_2D + offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load cc[j, i_val] for all j in this block  
    cc_offsets = offsets * LEN_2D + i_val
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Load bb[i_val, j] for all j in this block
    bb_offsets = i_val * LEN_2D + offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Compute aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
    result = aa_vals * cc_vals + bb_vals
    
    # Store back to aa[i_val, j]
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s1115_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i_val in range(LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s1115_kernel[grid](aa, bb, cc, i_val, LEN_2D, BLOCK_SIZE)