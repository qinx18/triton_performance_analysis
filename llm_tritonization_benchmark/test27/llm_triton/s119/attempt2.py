import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, diag, start_i, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offset within this diagonal
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate i and j values for elements in this diagonal
    i_vals = start_i + offset
    j_vals = diag - i_vals
    
    # Calculate end_i for this diagonal
    end_i = tl.minimum(diag, N - 1)
    
    # Mask for valid elements
    mask = (i_vals <= end_i) & (i_vals >= 1) & (j_vals >= 1) & (j_vals < N)
    
    # Calculate linear indices
    aa_indices = i_vals * N + j_vals
    bb_indices = i_vals * N + j_vals
    aa_dep_indices = (i_vals - 1) * N + (j_vals - 1)
    
    # Load values
    aa_dep_vals = tl.load(aa_ptr + aa_dep_indices, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = aa_dep_vals + bb_vals
    
    # Store results
    tl.store(aa_ptr + aa_indices, new_vals, mask=mask)

def s119_triton(aa, bb, len_2d):
    N = len_2d
    
    BLOCK_SIZE = 128
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):
        # Calculate range of i values for this diagonal
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N - 1)
        num_elements = end_i - start_i + 1
        
        if num_elements <= 0:
            continue
        
        # Calculate grid size
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        grid = (grid_size,)
        s119_kernel[grid](
            aa, bb, N, diag, start_i, BLOCK_SIZE
        )