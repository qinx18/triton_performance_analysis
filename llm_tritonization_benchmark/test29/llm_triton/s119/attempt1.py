import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        i_vals = diag - j_offsets
        
        # Mask for valid (i, j) coordinates
        mask = (i_vals >= 1) & (i_vals < N) & (j_offsets >= 1) & (j_offsets < N)
        
        # Calculate memory offsets
        read_offsets = (i_vals - 1) * N + (j_offsets - 1)
        write_offsets = i_vals * N + j_offsets
        bb_offsets = i_vals * N + j_offsets
        
        # Load data
        aa_prev = tl.load(aa_ptr + read_offsets, mask=mask, other=0.0)
        bb_curr = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
        
        # Compute
        result = aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + write_offsets, result, mask=mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    BLOCK_SIZE = 256
    
    # Ensure arrays are contiguous
    aa_flat = aa.contiguous().view(-1)
    bb_flat = bb.contiguous().view(-1)
    
    # Launch single kernel
    grid = (1,)
    s119_kernel[grid](aa_flat, bb_flat, N, BLOCK_SIZE)
    
    return aa