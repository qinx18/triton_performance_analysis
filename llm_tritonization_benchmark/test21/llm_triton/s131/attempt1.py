import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n - 1
    
    a_copy_vals = tl.load(a_copy_ptr + indices + 1, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_copy_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0]
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, n, BLOCK_SIZE=BLOCK_SIZE
    )