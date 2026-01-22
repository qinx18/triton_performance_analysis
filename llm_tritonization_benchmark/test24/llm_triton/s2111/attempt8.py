import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, diag, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate valid j range for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N - 1)
    
    # Get j indices for this block
    j_indices = start_j + block_start + offsets
    j_mask = j_indices <= end_j
    
    # Calculate corresponding i indices: i = diag - j
    i_indices = diag - j_indices
    i_mask = (i_indices >= 1) & (i_indices < N)
    
    # Combined mask
    mask = j_mask & i_mask
    
    # Load aa[j][i-1] values
    i_prev = i_indices - 1
    prev_i_offsets = j_indices * N + i_prev
    aa_j_i_prev = tl.load(aa_ptr + prev_i_offsets, mask=mask, other=0.0)
    
    # Load aa[j-1][i] values  
    j_prev = j_indices - 1
    prev_j_offsets = j_prev * N + i_indices
    aa_j_prev_i = tl.load(aa_ptr + prev_j_offsets, mask=mask, other=0.0)
    
    # Compute new values
    new_values = (aa_j_i_prev + aa_j_prev_i) / 1.9
    
    # Store results
    current_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + current_offsets, new_values, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1) 
        end_j = min(diag, N - 1)
        num_elements = end_j - start_j + 1
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s2111_kernel[grid](aa, diag, N, BLOCK_SIZE)