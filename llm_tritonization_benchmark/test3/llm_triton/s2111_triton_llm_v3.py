import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension (rows)
    j = tl.program_id(0) + 1
    
    # Sequential processing along i dimension due to WAR dependency
    for i in range(1, LEN_2D):
        # Load aa[j][i-1] and aa[j-1][i]
        left_idx = j * LEN_2D + (i - 1)
        above_idx = (j - 1) * LEN_2D + i
        current_idx = j * LEN_2D + i
        
        left_val = tl.load(aa_ptr + left_idx)
        above_val = tl.load(aa_ptr + above_idx)
        
        # Compute new value
        new_val = (left_val + above_val) / 1.9
        
        # Store result
        tl.store(aa_ptr + current_idx, new_val)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per row (j dimension)
    # Each thread processes one row sequentially
    grid = (LEN_2D - 1,)  # j from 1 to LEN_2D-1
    BLOCK_SIZE = 1
    
    s2111_kernel[grid](
        aa,
        LEN_2D,
        BLOCK_SIZE
    )
    
    return aa