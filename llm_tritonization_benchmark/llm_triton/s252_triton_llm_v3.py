import triton
import triton.language as tl
import torch

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    t = 0.0
    
    for i in range(0, n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s = b_val * c_val
        result = s + t
        tl.store(a_ptr + i, result)
        t = s

def s252_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s252_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a