import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one row at a time due to dependencies
    # Each program instance handles one j value
    j = tl.program_id(0) + 1  # Start from j=1
    
    if j >= LEN_2D:
        return
    
    # Process row j sequentially since aa[j][i] depends on aa[j][i-1]
    for i in range(1, LEN_2D):
        # Load aa[j][i-1]
        prev_offset = j * LEN_2D + (i - 1)
        prev_val = tl.load(aa_ptr + prev_offset)
        
        # Load aa[j-1][i]
        above_offset = (j - 1) * LEN_2D + i
        above_val = tl.load(aa_ptr + above_offset)
        
        # Compute new value
        new_val = (prev_val + above_val) / 1.9
        
        # Store aa[j][i]
        curr_offset = j * LEN_2D + i
        tl.store(aa_ptr + curr_offset, new_val)

def s2111_triton(aa):
    LEN_2D = aa.size(0)
    
    # Launch kernel with one program per row (excluding first row j=0)
    grid = (LEN_2D - 1,)
    s2111_kernel[grid](aa, LEN_2D, BLOCK_SIZE=256)