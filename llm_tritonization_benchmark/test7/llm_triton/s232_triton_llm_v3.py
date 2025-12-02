import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    j = pid_j + 1
    
    if j >= LEN_2D:
        return
    
    # Process all i values for this j sequentially (due to dependency)
    for i in range(1, j + 1):
        # Load aa[j][i-1] and aa[j][i]
        prev_val = tl.load(aa_ptr + j * LEN_2D + (i - 1))
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Compute aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
        new_val = prev_val * prev_val + bb_val
        
        # Store result
        tl.store(aa_ptr + j * LEN_2D + i, new_val)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with one thread block per j value
    grid = (LEN_2D - 1,)  # j ranges from 1 to LEN_2D-1
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa