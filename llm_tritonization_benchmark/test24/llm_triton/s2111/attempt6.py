import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    diag_idx = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    
    # Calculate diagonal value: diag = 2 + diag_idx
    diag = 2 + diag_idx
    
    # Calculate range of j values for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N)
    
    # Calculate actual j for this block
    j_offset = start_j + block_start
    offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_offset + offsets
    
    # Mask for valid j values
    j_mask = (j_indices < end_j) & (j_indices >= start_j)
    
    # Calculate corresponding i values: i = diag - j
    i_indices = diag - j_indices
    i_mask = (i_indices >= 1) & (i_indices < N)
    
    # Combined mask
    mask = j_mask & i_mask
    
    # Load aa[j-1, i] values
    j_prev_indices = j_indices - 1
    prev_j_offsets = j_prev_indices * N + i_indices
    aa_prev_j = tl.load(aa_ptr + prev_j_offsets, mask=mask, other=0.0)
    
    # Load aa[j, i-1] values
    i_prev_indices = i_indices - 1
    prev_i_offsets = j_indices * N + i_prev_indices
    aa_prev_i = tl.load(aa_ptr + prev_i_offsets, mask=mask, other=0.0)
    
    # Compute new values
    new_values = (aa_prev_j + aa_prev_i) / 1.9
    
    # Store results
    current_offsets = j_indices * N + i_indices
    tl.store(aa_ptr + current_offsets, new_values, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process each diagonal sequentially
    max_diag = 2 * N - 3  # Maximum diagonal index
    
    for diag_idx in range(max_diag):
        diag = 2 + diag_idx
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        num_elements = end_j - start_j
        
        if num_elements > 0:
            grid = (1, triton.cdiv(num_elements, BLOCK_SIZE))
            s2111_kernel[grid](aa, N, BLOCK_SIZE)