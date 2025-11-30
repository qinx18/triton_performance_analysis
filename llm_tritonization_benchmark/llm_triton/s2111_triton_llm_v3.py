import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(
    aa_ptr,
    aa_copy_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Calculate which j we're working on
    j = (pid // ((LEN_2D - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE)) + 1
    
    # Calculate block index within this j
    block_idx = pid % ((LEN_2D - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Calculate i range for this block
    i_start = block_idx * BLOCK_SIZE + 1
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    i_mask = (i_offsets < LEN_2D) & (j < LEN_2D)
    
    # Calculate memory offsets
    current_offsets = j * LEN_2D + i_offsets
    left_offsets = j * LEN_2D + (i_offsets - 1)
    top_offsets = (j - 1) * LEN_2D + i_offsets
    
    # Load values from the read-only copy
    left_vals = tl.load(aa_copy_ptr + left_offsets, mask=i_mask)
    top_vals = tl.load(aa_copy_ptr + top_offsets, mask=i_mask)
    
    # Compute result
    result = (left_vals + top_vals) / 1.9
    
    # Store to original array
    tl.store(aa_ptr + current_offsets, result, mask=i_mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate grid size - need to process each j sequentially
    blocks_per_j = (LEN_2D - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each j sequentially to maintain dependencies
    for j in range(1, LEN_2D):
        grid = (blocks_per_j,)
        
        # Create a temporary view for this j only
        j_start = j
        j_end = j + 1
        
        s2111_kernel[grid](
            aa,
            aa_copy,
            LEN_2D,
            BLOCK_SIZE,
        )
        
        # Update the copy with the newly computed values for this j
        # This ensures the next iteration sees the updated values
        aa_copy[j:j+1] = aa[j:j+1]