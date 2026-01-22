import triton
import triton.language as tl
import torch

@triton.jit
def s431_kernel(a_ptr, a_copy_ptr, b_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    read_offsets = offsets + k
    read_mask = mask & (read_offsets < n)
    
    a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s431_triton(a, b, k):
    n = a.shape[0]
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s431_kernel[grid](
        a, a_copy, b, k, n,
        BLOCK_SIZE=BLOCK_SIZE
    )