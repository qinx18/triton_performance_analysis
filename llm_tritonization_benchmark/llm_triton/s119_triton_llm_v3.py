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
    # Get program ID for j dimension (parallelized)
    j_block_id = tl.program_id(0)
    j_start = j_block_id * BLOCK_SIZE + 1
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        # Calculate j offsets for current block
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Calculate array indices
        current_indices = i * LEN_2D + j_offsets
        read_indices = (i - 1) * LEN_2D + (j_offsets - 1)
        
        # Load values from read-only copy and bb array
        aa_vals = tl.load(aa_copy_ptr + read_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + current_indices, mask=j_mask)
        
        # Compute result
        result = aa_vals + bb_vals
        
        # Store result to original aa array
        tl.store(aa_ptr + current_indices, result, mask=j_mask)

def s119_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate grid dimensions
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
    grid = (num_blocks,)
    
    # Launch kernel
    s119_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )