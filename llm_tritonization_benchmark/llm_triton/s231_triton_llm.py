import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s231: aa[j][i] = aa[j-1][i] + bb[j][i]
    Each block processes one column (fixed i) across all rows j.
    """
    # Get column index
    col_idx = tl.program_id(0)
    
    # Process sequentially within each column to maintain data dependency
    # aa[j][i] depends on aa[j-1][i], so we must process j=1,2,3... in order
    for j in range(1, LEN_2D):
        # Calculate memory addresses for current and previous row
        curr_addr = aa_ptr + j * LEN_2D + col_idx
        prev_addr = aa_ptr + (j - 1) * LEN_2D + col_idx
        bb_addr = bb_ptr + j * LEN_2D + col_idx
        
        # Mask to ensure we don't access out of bounds
        mask = col_idx < LEN_2D
        
        # Load values with masking
        aa_prev = tl.load(prev_addr, mask=mask)
        bb_curr = tl.load(bb_addr, mask=mask)
        
        # Compute and store result
        result = aa_prev + bb_curr
        tl.store(curr_addr, result, mask=mask)

def s231_triton(aa, bb):
    """
    Triton implementation of TSVC s231.
    
    Each thread block processes one column, ensuring data dependencies
    within columns are handled correctly through sequential processing.
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Launch one thread block per column
    # Each block processes all rows for its column sequentially
    BLOCK_SIZE = 128
    grid = (LEN_2D,)
    
    s231_kernel[grid](
        aa, bb, LEN_2D, BLOCK_SIZE
    )
    
    return aa