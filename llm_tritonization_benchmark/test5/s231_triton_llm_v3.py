import triton
import triton.language as tl
import torch

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column index
    col_id = tl.program_id(0)
    
    if col_id >= LEN_2D:
        return
    
    # Process column col_id sequentially due to data dependency
    for j in range(1, LEN_2D):
        # Calculate memory offsets
        curr_offset = j * LEN_2D + col_id
        prev_offset = (j - 1) * LEN_2D + col_id
        
        # Load values
        aa_prev = tl.load(aa_ptr + prev_offset)
        bb_curr = tl.load(bb_ptr + curr_offset)
        
        # Compute and store
        result = aa_prev + bb_curr
        tl.store(aa_ptr + curr_offset, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per column
    grid = (LEN_2D,)
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=1
    )