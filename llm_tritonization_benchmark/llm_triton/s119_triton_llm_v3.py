import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current i value (sequential)
    i = tl.program_id(0) + 1
    
    # Get the starting j position for this block
    j_block_start = tl.program_id(1) * BLOCK_SIZE + 1
    
    # Create offsets for the j dimension
    j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate memory offsets for current position aa[i][j]
    current_offsets = i * LEN_2D + j_offsets
    
    # Calculate memory offsets for reading aa[i-1][j-1]
    read_offsets = (i - 1) * LEN_2D + (j_offsets - 1)
    
    # Load values from read-only copy of aa and from bb
    aa_vals = tl.load(aa_copy_ptr + read_offsets, mask=j_mask)
    bb_vals = tl.load(bb_ptr + current_offsets, mask=j_mask)
    
    # Compute the result
    result = aa_vals + bb_vals
    
    # Store to the original aa array
    tl.store(aa_ptr + current_offsets, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy of aa
    aa_copy = aa.clone()
    
    # Calculate grid dimensions
    # i dimension: sequential from 1 to LEN_2D-1
    # j dimension: parallel blocks
    grid_i = LEN_2D - 1
    grid_j = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
    
    # Launch kernel with i-sequential, j-parallel strategy
    for i_offset in range(grid_i):
        s119_kernel[(1, grid_j)](
            aa,
            aa_copy,
            bb,
            LEN_2D,
            BLOCK_SIZE,
        )
        # Update the copy after each i iteration to maintain dependencies
        if i_offset < grid_i - 1:
            aa_copy = aa.clone()