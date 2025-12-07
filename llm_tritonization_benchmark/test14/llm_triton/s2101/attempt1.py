import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < LEN_2D
    
    # Calculate diagonal indices (i*LEN_2D + i for aa[i][i])
    diagonal_indices = offsets * LEN_2D + offsets
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diagonal_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + diagonal_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + diagonal_indices, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + diagonal_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )