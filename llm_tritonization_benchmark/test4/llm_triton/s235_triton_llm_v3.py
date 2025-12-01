import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(a_ptr, b_ptr, c_ptr, aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Load and update a[i]
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    a_val = a_val + b_val * c_val
    tl.store(a_ptr + i, a_val)
    
    # Inner loop: sequential dependency on j-1
    for j in range(1, LEN_2D):
        aa_prev = tl.load(aa_ptr + (j-1) * LEN_2D + i)
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        aa_new = aa_prev + bb_val * a_val
        tl.store(aa_ptr + j * LEN_2D + i, aa_new)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s235_kernel[grid](
        a, b, c, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )