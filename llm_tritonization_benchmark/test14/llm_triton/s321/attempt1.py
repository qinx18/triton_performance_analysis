import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    for i in range(1, n_elements):
        a_val = tl.load(a_ptr + i)
        a_prev = tl.load(a_ptr + i - 1)
        b_val = tl.load(b_ptr + i)
        result = a_val + a_prev * b_val
        tl.store(a_ptr + i, result)

def s321_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 128
    
    s321_kernel[(1,)](
        a, b, n_elements, BLOCK_SIZE
    )