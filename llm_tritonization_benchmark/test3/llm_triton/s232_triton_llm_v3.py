import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one row at a time due to the triangular dependency
    j = tl.program_id(0) + 1
    
    if j >= LEN_2D:
        return
    
    # Process elements in the triangular region for this row
    for i in range(1, j + 1):
        # Load aa[j][i-1], aa[j][i], and bb[j][i]
        aa_prev = tl.load(aa_ptr + j * LEN_2D + (i - 1))
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        
        # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
        result = aa_prev * aa_prev + bb_val
        
        # Store result
        tl.store(aa_ptr + j * LEN_2D + i, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per row (excluding row 0)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 256
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )