import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Calculate bounds for this diagonal
    start_i = tl.maximum(1, diag - N + 1)
    end_i = tl.minimum(diag - 1, N - 1)
    
    # Calculate offsets within this diagonal
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Convert diagonal-relative indices to (i, j) coordinates
    i_vals = start_i + indices
    j_vals = diag - i_vals
    
    # Create mask for valid elements
    mask = (indices <= (end_i - start_i)) & (i_vals < N) & (j_vals < N) & (i_vals >= 1) & (j_vals >= 1)
    
    # Calculate linear indices
    aa_indices = i_vals * N + j_vals
    bb_indices = i_vals * N + j_vals
    aa_dep_indices = (i_vals - 1) * N + (j_vals - 1)
    
    # Load values
    aa_dep_vals = tl.load(aa_ptr + aa_dep_indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Compute result
    result = aa_dep_vals + bb_vals
    
    # Store result
    tl.store(aa_ptr + aa_indices, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)
        num_elements = end_i - start_i + 1
        
        if num_elements > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s119_kernel[grid](
                aa, bb, diag, N,
                BLOCK_SIZE=BLOCK_SIZE
            )