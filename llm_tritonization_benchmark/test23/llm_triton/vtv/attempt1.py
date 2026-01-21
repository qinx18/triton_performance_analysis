import triton
import triton.language as tl
import torch

@triton.jit
def vtv_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    result = a_vals * b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def vtv_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vtv_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)