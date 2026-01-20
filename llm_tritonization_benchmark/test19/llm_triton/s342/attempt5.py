import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        j = -1
        for i in range(n):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                j += 1
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    
    BLOCK_SIZE = 1
    grid = (1,)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )