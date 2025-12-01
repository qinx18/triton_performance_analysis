import triton
import triton.language as tl
import torch

def f(x, y):
    return x * y

@triton.jit
def s4121_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    result = b_vals * c_vals
    updated_a = a_vals + result
    
    tl.store(a_ptr + offsets, updated_a, mask=mask)

def s4121_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4121_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a