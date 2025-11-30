import torch
import triton
import triton.language as tl

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Calculate diagonal indices (i, i) -> i * LEN_2D + i
    diagonal_indices = offsets * LEN_2D + offsets
    
    # Load diagonal elements
    bb_diag = tl.load(bb_ptr + diagonal_indices, mask=mask)
    cc_diag = tl.load(cc_ptr + diagonal_indices, mask=mask)
    aa_diag = tl.load(aa_ptr + diagonal_indices, mask=mask)
    
    # Perform computation: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_diag + bb_diag * cc_diag
    
    # Store result
    tl.store(aa_ptr + diagonal_indices, result, mask=mask)

def s2101_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )