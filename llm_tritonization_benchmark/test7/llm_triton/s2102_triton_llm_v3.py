import torch
import triton
import triton.language as tl

@triton.jit
def s2102_kernel(
    aa,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for the current row
    row_id = tl.program_id(0)
    
    # Define column offsets once at kernel start
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Zero out the entire row
    for col_start in range(0, LEN_2D, BLOCK_SIZE):
        current_col_offsets = col_start + col_offsets
        col_mask = current_col_offsets < LEN_2D
        
        # Calculate pointer to aa[current_col_offsets][row_id]
        ptr = aa + current_col_offsets * LEN_2D + row_id
        
        # Store zeros
        tl.store(ptr, 0.0, mask=col_mask)
    
    # Set diagonal element to 1.0
    diagonal_ptr = aa + row_id * LEN_2D + row_id
    tl.store(diagonal_ptr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread block per row
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )