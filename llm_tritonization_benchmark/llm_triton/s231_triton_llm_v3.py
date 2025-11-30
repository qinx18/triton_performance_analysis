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
    # Get the column index (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process column i sequentially from j=1 to LEN_2D-1
    for j in range(1, LEN_2D):
        # Calculate offsets for current and previous row in column i
        curr_offset = j * LEN_2D + i
        prev_offset = (j - 1) * LEN_2D + i
        
        # Load values
        aa_prev = tl.load(aa_copy_ptr + prev_offset)
        bb_curr = tl.load(bb_ptr + curr_offset)
        
        # Compute result
        result = aa_prev + bb_curr
        
        # Store result
        tl.store(aa_ptr + curr_offset, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy to handle WAR dependency
    aa_copy = aa.clone()
    
    # Launch kernel with one thread per column (i dimension)
    grid = (LEN_2D,)
    
    s231_kernel[grid](
        aa,
        aa_copy,
        bb,
        LEN_2D,
        256,
    )