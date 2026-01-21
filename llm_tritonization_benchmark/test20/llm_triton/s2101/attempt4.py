import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this thread block
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases
    mask = offsets < N
    
    # Calculate diagonal indices (i*N + i for aa[i][i])
    diagonal_offsets = offsets * (N + 1)
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + diagonal_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + diagonal_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + diagonal_offsets, mask=mask)
    
    # Perform computation: aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result back to aa
    tl.store(aa_ptr + diagonal_offsets, result, mask=mask)

def s2101_triton(aa, bb, cc):
    N = aa.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s2101_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)