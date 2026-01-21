import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n):
    pid = tl.program_id(axis=0)
    if pid > 0:
        return
    
    for i in range(2, n):
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i = tl.load(a_ptr + i)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    grid = (1,)
    s322_kernel[grid](a, b, c, n)
    
    return a