import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially inside kernel
    for diag in range(2, 2 * N):
        i_vals = i_offsets
        j_vals = diag - i_vals
        
        # Mask for valid (j, i) pairs within bounds
        mask = (j_vals >= 1) & (j_vals < N) & (i_vals >= 1) & (i_vals < N)
        
        # Calculate indices for current position, left neighbor, and top neighbor
        curr_idx = j_vals * N + i_vals
        left_idx = j_vals * N + (i_vals - 1)
        top_idx = (j_vals - 1) * N + i_vals
        
        # Load values with proper masking
        left_vals = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
        top_vals = tl.load(aa_ptr + top_idx, mask=mask, other=0.0)
        
        # Compute new values: (aa[j][i-1] + aa[j-1][i]) / 1.9
        new_vals = (left_vals + top_vals) * (1.0 / 1.9)
        
        # Store results
        tl.store(aa_ptr + curr_idx, new_vals, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (1,)
    s2111_kernel[grid](aa, N, BLOCK_SIZE)