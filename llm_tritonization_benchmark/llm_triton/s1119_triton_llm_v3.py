import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension (sequential processing)
    i = tl.program_id(0) + 1  # Start from i=1
    
    # Get block start for j dimension (parallel processing)
    j_block_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create offsets for j dimension
    j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate memory offsets for current row (i) and previous row (i-1)
    curr_row_offset = i * LEN_2D + j_offsets
    prev_row_offset = (i - 1) * LEN_2D + j_offsets
    
    # Load from read-only copy for aa[i-1][j] and original bb for bb[i][j]
    aa_prev_vals = tl.load(aa_copy_ptr + prev_row_offset, mask=j_mask)
    bb_vals = tl.load(bb_ptr + curr_row_offset, mask=j_mask)
    
    # Compute result
    result = aa_prev_vals + bb_vals
    
    # Store to original aa array
    tl.store(aa_ptr + curr_row_offset, result, mask=j_mask)

def s1119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    # i dimension: sequential processing from 1 to LEN_2D-1
    # j dimension: parallel blocks
    i_blocks = LEN_2D - 1  # Process i from 1 to LEN_2D-1
    j_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Process each i sequentially to maintain dependencies
    for i_offset in range(i_blocks):
        grid = (1, j_blocks)  # Single i value per launch, multiple j blocks
        
        s1119_kernel[grid](
            aa,
            aa_copy,
            bb,
            LEN_2D,
            BLOCK_SIZE,
        )
        
        # Update the copy for next iteration to reflect current state
        if i_offset < i_blocks - 1:  # Not the last iteration
            aa_copy = aa.clone()