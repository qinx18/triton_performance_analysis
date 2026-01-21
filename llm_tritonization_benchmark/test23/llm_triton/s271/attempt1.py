import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    condition = b_vals > 0.0
    
    valid_mask = mask & condition
    
    a_vals = tl.load(a_ptr + offsets, mask=valid_mask)
    c_vals = tl.load(c_ptr + offsets, mask=valid_mask)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, result, mask=valid_mask)

def s271_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)