import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = (indices < n - 1) & (k > 0)
    read_indices = indices + k
    read_mask = mask & (read_indices < n)
    
    a_vals = tl.load(a_copy_ptr + read_indices, mask=read_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    n = a.shape[0]
    
    if k <= 0:
        return
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_copy, b, c, k, n, BLOCK_SIZE
    )