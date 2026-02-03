import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially inside kernel
    for diag in range(2, 2 * N):
        j_vals = diag - i_offsets
        i_vals = i_offsets
        
        # Mask for valid (j, i) pairs
        mask = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
        
        # Calculate indices for current position, left neighbor, and top neighbor
        curr_idx = j_vals * N + i_vals
        left_idx = j_vals * N + (i_vals - 1)
        top_idx = (j_vals - 1) * N + i_vals
        
        # Load values
        left_vals = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
        top_vals = tl.load(aa_ptr + top_idx, mask=mask, other=0.0)
        
        # Compute new values
        new_vals = (left_vals + top_vals) / 1.9
        
        # Store results
        tl.store(aa_ptr + curr_idx, new_vals, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 512
    
    grid = (1,)
    s2111_kernel[grid](aa, N, BLOCK_SIZE)