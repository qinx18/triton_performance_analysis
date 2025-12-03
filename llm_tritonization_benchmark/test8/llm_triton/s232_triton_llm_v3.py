import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr):
    j = tl.program_id(0) + 1
    
    if j < LEN_2D:
        for i in range(1, j + 1):
            aa_prev_val = tl.load(aa_ptr + j * LEN_2D + (i - 1))
            bb_val = tl.load(bb_ptr + j * LEN_2D + i)
            result = aa_prev_val * aa_prev_val + bb_val
            tl.store(aa_ptr + j * LEN_2D + i, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    grid = (LEN_2D - 1,)
    
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D
    )