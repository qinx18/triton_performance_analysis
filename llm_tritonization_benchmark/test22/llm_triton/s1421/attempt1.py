import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(b_ptr, xx_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n
    
    xx_vals = tl.load(xx_ptr + block_start + offsets, mask=mask)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    result = xx_vals + a_vals
    
    tl.store(b_ptr + block_start + offsets, result, mask=mask)

def s1421_triton(a, b, xx):
    n = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1421_kernel[grid](b, xx, a, n, BLOCK_SIZE=BLOCK_SIZE)