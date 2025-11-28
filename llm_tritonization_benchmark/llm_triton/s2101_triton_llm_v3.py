import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate the range of diagonal elements this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Calculate diagonal offsets (i*LEN_2D + i for element [i][i])
    diag_offsets = offsets * LEN_2D + offsets
    
    # Load diagonal elements with masking
    aa_vals = tl.load(aa_ptr + diag_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diag_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diag_offsets, mask=mask)
    
    # Perform computation: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result back to diagonal elements
    tl.store(aa_ptr + diag_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Choose block size - power of 2, reasonable for diagonal access
    BLOCK_SIZE = 128
    
    # Calculate number of blocks needed
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    # Launch kernel
    s2101_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE,
    )
    
    return aa