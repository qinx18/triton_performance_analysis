import torch
import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Sequential dependency in j dimension - must process sequentially
    for j in range(1, LEN_2D):
        # Load aa[j-1][i] 
        prev_val = tl.load(aa_ptr + (j - 1) * LEN_2D + i)
        
        # Load bb[j][i]
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_val + bb_val
        
        # Store result
        tl.store(aa_ptr + j * LEN_2D + i, result)

def s231_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per i
    grid = (LEN_2D,)
    BLOCK_SIZE = 1
    
    s231_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )