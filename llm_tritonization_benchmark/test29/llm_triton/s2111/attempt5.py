import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    tid = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        # For anti-diagonal diag, we have j + i = diag
        # So j = diag - i
        i_vals = tid
        j_vals = diag - i_vals
        
        # Mask for valid (j, i) pairs within bounds
        valid_mask = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N) & (tid < BLOCK_SIZE)
        
        # Calculate linear indices for row-major order (j*N + i)
        current_idx = j_vals * N + i_vals
        left_idx = j_vals * N + (i_vals - 1)
        above_idx = (j_vals - 1) * N + i_vals
        
        # Load values with mask
        left_vals = tl.load(aa_ptr + left_idx, mask=valid_mask, other=0.0)
        above_vals = tl.load(aa_ptr + above_idx, mask=valid_mask, other=0.0)
        
        # Compute new values: (left + above) / 1.9
        new_vals = (left_vals + above_vals) / 1.9
        
        # Store results
        tl.store(aa_ptr + current_idx, new_vals, mask=valid_mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 1024
    
    # Flatten the tensor to 1D for easier indexing
    aa_flat = aa.view(-1)
    
    # Single kernel launch
    grid = (1,)
    s2111_kernel[grid](aa_flat, N, BLOCK_SIZE)