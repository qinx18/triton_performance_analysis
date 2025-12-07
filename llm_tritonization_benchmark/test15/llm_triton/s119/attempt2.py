import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag_val, start_i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    element_indices = block_start + offsets
    
    # Calculate i values for this diagonal
    i_vals = start_i + element_indices
    j_vals = diag_val - i_vals
    
    # Create mask for valid elements
    mask = (i_vals < LEN_2D) & (j_vals >= 1) & (j_vals < LEN_2D) & (i_vals >= 1)
    
    # Calculate memory offsets
    aa_offsets = i_vals * LEN_2D + j_vals
    aa_prev_offsets = (i_vals - 1) * LEN_2D + (j_vals - 1)
    bb_offsets = i_vals * LEN_2D + j_vals
    
    # Load data
    aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Compute and store
    result = aa_prev_vals + bb_vals
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * LEN_2D - 1):  # diag = i + j
        # Calculate range of i values for this diagonal
        start_i = max(1, diag - LEN_2D + 1)
        end_i = min(diag, LEN_2D - 1)
        
        # Skip if no valid elements
        if start_i > end_i:
            continue
        
        num_elements = end_i - start_i + 1
        
        # Calculate grid size
        grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        s119_kernel[(grid_size,)](
            aa, bb, diag, start_i, LEN_2D, BLOCK_SIZE
        )