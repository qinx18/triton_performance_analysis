import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for j dimension
    j = tl.program_id(0) + 1
    
    if j >= LEN_2D:
        return
    
    # Sequential processing for i dimension due to dependencies
    for i in range(1, j + 1):
        # Calculate offsets for 2D arrays (row-major order)
        current_offset = j * LEN_2D + i
        prev_offset = j * LEN_2D + (i - 1)
        
        # Load values
        aa_prev = tl.load(aa_copy_ptr + prev_offset)
        bb_val = tl.load(bb_ptr + current_offset)
        
        # Compute result
        result = aa_prev * aa_prev + bb_val
        
        # Store result
        tl.store(aa_ptr + current_offset, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Grid configuration - one program per j value
    grid = (LEN_2D - 1,)
    
    # Launch kernel
    s232_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D,
        BLOCK_SIZE=256,
    )