import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current block's starting position for the i dimension
    block_id = tl.program_id(0)
    i = block_id
    
    if i >= LEN_2D:
        return
    
    # Process j dimension sequentially from 1 to LEN_2D-1
    for j in range(1, LEN_2D):
        # Current position: aa[j][i]
        current_offset = j * LEN_2D + i
        # Previous position: aa[j-1][i]
        prev_offset = (j - 1) * LEN_2D + i
        
        # Load from the read-only copy and bb array
        aa_prev_val = tl.load(aa_copy_ptr + prev_offset)
        bb_val = tl.load(bb_ptr + current_offset)
        
        # Compute result
        result = aa_prev_val + bb_val
        
        # Store to original aa array
        tl.store(aa_ptr + current_offset, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependency
    aa_copy = aa.clone()
    
    # Grid size: one block per column (i dimension)
    grid = (LEN_2D,)
    
    # Launch kernel
    s231_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=1,
    )