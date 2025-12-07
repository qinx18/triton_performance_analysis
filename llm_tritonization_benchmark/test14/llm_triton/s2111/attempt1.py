import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get diagonal index
    diag_idx = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    
    # Calculate actual diagonal value (diag = i + j)
    diag = diag_idx + 2
    
    # Calculate range of j values for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N)
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = start_j + block_start + offsets
    
    # Mask for valid elements
    mask = (j_indices < end_j) & (j_indices >= start_j)
    
    # Calculate corresponding i indices
    i_indices = diag - j_indices
    
    # Additional mask for valid i range
    mask = mask & (i_indices >= 1) & (i_indices < N)
    
    # Load aa[j][i-1] values
    left_indices = j_indices * N + (i_indices - 1)
    left_vals = tl.load(aa_ptr + left_indices, mask=mask, other=0.0)
    
    # Load aa[j-1][i] values  
    up_indices = (j_indices - 1) * N + i_indices
    up_vals = tl.load(aa_ptr + up_indices, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (left_vals + up_vals) / 1.9
    
    # Store results
    curr_indices = j_indices * N + i_indices
    tl.store(aa_ptr + curr_indices, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2*N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        num_elements = end_j - start_j
        
        if num_elements <= 0:
            continue
            
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        grid = (1, num_blocks)
        
        s2111_kernel[grid](
            aa,
            N,
            BLOCK_SIZE
        )
    
    return aa