import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for i dimension (columns)
    i = tl.program_id(0)
    
    # Process column i sequentially for j from 1 to LEN_2D-1
    for j in range(1, LEN_2D):
        # Calculate offsets for current and previous row
        current_offset = j * LEN_2D + i
        prev_offset = (j - 1) * LEN_2D + i
        
        # Load values from read-only copy and bb array
        aa_prev_val = tl.load(aa_copy_ptr + prev_offset)
        bb_val = tl.load(bb_ptr + current_offset)
        
        # Compute result
        result = aa_prev_val + bb_val
        
        # Store to original aa array
        tl.store(aa_ptr + current_offset, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Launch kernel with one thread per column (i dimension)
    grid = (LEN_2D,)
    BLOCK_SIZE = 256
    
    s231_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D,
        BLOCK_SIZE,
    )