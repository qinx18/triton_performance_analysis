import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for diagonal processing
    diag_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Calculate actual diagonal value (diag = i + j)
    diag = 2 + diag_id
    
    # Calculate range of i values for this diagonal
    start_i = max(1, diag - LEN_2D + 1)
    end_i = min(diag, LEN_2D)
    num_elements = end_i - start_i
    
    # Skip if no valid elements
    if num_elements <= 0:
        return
    
    # Calculate block range
    block_start = block_id * BLOCK_SIZE
    if block_start >= num_elements:
        return
    
    # Create offset vector for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    element_offsets = block_start + offsets
    
    # Calculate actual i, j coordinates
    i_vals = start_i + element_offsets
    j_vals = diag - i_vals
    
    # Create mask for valid elements
    mask = (element_offsets < num_elements) & (i_vals < LEN_2D) & (j_vals < LEN_2D) & (i_vals >= 1) & (j_vals >= 1)
    
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
    num_diagonals = 2 * LEN_2D - 3  # from diag=2 to diag=2*LEN_2D-2
    
    for diag_offset in range(num_diagonals):
        diag = 2 + diag_offset
        
        # Calculate range of i values for this diagonal
        start_i = max(1, diag - LEN_2D + 1)
        end_i = min(diag, LEN_2D)
        num_elements = end_i - start_i
        
        if num_elements <= 0:
            continue
        
        # Calculate grid size for this diagonal
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        
        # Launch kernel for this diagonal
        grid = (1, num_blocks)
        s119_kernel[grid](
            aa, bb, LEN_2D, BLOCK_SIZE
        )