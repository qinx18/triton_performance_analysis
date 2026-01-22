import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, start_j: tl.constexpr, end_j: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the block of j indices for this anti-diagonal
    block_idx = tl.program_id(0)
    
    # Calculate j indices for this block
    j_start = start_j + block_idx * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j indices
    j_mask = (j_offsets >= start_j) & (j_offsets < end_j)
    
    # Calculate corresponding i indices (i = diag - j)
    i_offsets = diag - j_offsets
    
    # Additional mask for valid i indices
    valid_mask = j_mask & (i_offsets >= 1) & (i_offsets < N) & (j_offsets >= 1) & (j_offsets < N)
    
    # Calculate linear indices for current positions
    current_indices = j_offsets * N + i_offsets
    
    # Calculate linear indices for dependencies
    left_indices = j_offsets * N + (i_offsets - 1)  # aa[j][i-1]
    up_indices = (j_offsets - 1) * N + i_offsets    # aa[j-1][i]
    
    # Load values with masking
    left_vals = tl.load(aa_ptr + left_indices, mask=valid_mask, other=0.0)
    up_vals = tl.load(aa_ptr + up_indices, mask=valid_mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    tl.store(aa_ptr + current_indices, new_vals, mask=valid_mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):  # diag = i + j, starting from 2
        # Calculate range of j indices for this diagonal
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        if start_j >= end_j:
            continue
            
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, 
            N=N, 
            diag=diag, 
            start_j=start_j, 
            end_j=end_j, 
            BLOCK_SIZE=BLOCK_SIZE
        )