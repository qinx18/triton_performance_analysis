import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets * inc
    
    mask = i_offsets < (n - 1)
    
    a_copy_vals = tl.load(a_copy_ptr + i_offsets + inc, mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    result = a_copy_vals + b_vals
    
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n = a.shape[0]
    
    a_copy = a.clone()
    
    total_elements = (n - 1 + inc - 1) // inc
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n, BLOCK_SIZE=BLOCK_SIZE
    )