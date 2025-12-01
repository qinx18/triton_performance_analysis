import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one j value (row) at a time
    j = tl.program_id(0) + 1
    
    if j >= LEN_2D:
        return
    
    # Sequential computation within each row
    for i in range(1, LEN_2D):
        # Load aa[j][i-1] (left neighbor)
        left_val = tl.load(aa_ptr + j * LEN_2D + (i - 1))
        
        # Load aa[j-1][i] (top neighbor)
        top_val = tl.load(aa_ptr + (j - 1) * LEN_2D + i)
        
        # Compute new value
        new_val = (left_val + top_val) / 1.9
        
        # Store result
        tl.store(aa_ptr + j * LEN_2D + i, new_val)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread block per row (j dimension)
    # Each thread block handles one row sequentially
    grid = (LEN_2D - 1,)  # j from 1 to LEN_2D-1
    
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=256
    )