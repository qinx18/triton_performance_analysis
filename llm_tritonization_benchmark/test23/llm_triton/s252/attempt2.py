import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process sequentially element by element
    t = 0.0
    
    for i in range(N):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s = b_val * c_val
        a_val = s + t
        tl.store(a_ptr + i, a_val)
        t = s

def s252_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, N,
        BLOCK_SIZE=BLOCK_SIZE
    )