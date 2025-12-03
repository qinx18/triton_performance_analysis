import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Load and update a[i] += b[i] * c[i]
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    a_val = tl.load(a_ptr + i)
    a_val = a_val + b_val * c_val
    tl.store(a_ptr + i, a_val)
    
    # Inner loop: for j = 1 to LEN_2D-1
    # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
    for j in range(1, LEN_2D):
        aa_prev = tl.load(aa_ptr + (j-1) * LEN_2D + i)
        bb_val = tl.load(bb_ptr + j * LEN_2D + i)
        aa_new = aa_prev + bb_val * a_val
        tl.store(aa_ptr + j * LEN_2D + i, aa_new)

def s235_triton(a, aa, b, bb, c):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s235_kernel[grid](
        a, aa, b, bb, c,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )