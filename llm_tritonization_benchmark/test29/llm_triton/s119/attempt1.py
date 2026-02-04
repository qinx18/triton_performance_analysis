import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially inside kernel
    for diag in range(2, 2 * N):
        i_vals = diag - j_offsets
        
        # Mask for valid indices
        mask = (i_vals >= 1) & (i_vals < N) & (j_offsets >= 1) & (j_offsets < N)
        
        # Calculate linear indices for aa[i,j], aa[i-1,j-1], and bb[i,j]
        aa_current_idx = i_vals * N + j_offsets
        aa_prev_idx = (i_vals - 1) * N + (j_offsets - 1)
        bb_idx = i_vals * N + j_offsets
        
        # Load values
        aa_prev = tl.load(aa_ptr + aa_prev_idx, mask=mask, other=0.0)
        bb_val = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
        
        # Compute new values
        result = aa_prev + bb_val
        
        # Store results
        tl.store(aa_ptr + aa_current_idx, result, mask=mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    BLOCK_SIZE = 256
    
    # Single kernel launch with one block
    grid = (1,)
    s119_kernel[grid](aa, bb, N, BLOCK_SIZE)