import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Calculate diagonal indices (i * n + i for 2D array flattened)
    diagonal_offsets = offsets * n + offsets
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diagonal_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diagonal_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diagonal_offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store back to aa
    tl.store(aa_ptr + diagonal_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa