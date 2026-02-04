import triton
import triton.language as tl
import torch

@triton.jit
def s421_kernel_copy(xx_ptr, yy_ptr, n):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    mask = idx < n
    
    xx_vals = tl.load(xx_ptr + idx, mask=mask)
    tl.store(yy_ptr + idx, xx_vals, mask=mask)

@triton.jit
def s421_kernel_compute(xx_ptr, yy_ptr, a_ptr, n):
    pid = tl.program_id(0)
    BLOCK_SIZE = 256
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    mask = idx < (n - 1)
    
    yy_next = tl.load(yy_ptr + idx + 1, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    result = yy_next + a_vals
    tl.store(xx_ptr + idx, result, mask=mask)

def s421_triton(a, xx, yy):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid_full = (triton.cdiv(n, BLOCK_SIZE),)
    grid_partial = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s421_kernel_copy[grid_full](xx, yy, n)
    s421_kernel_compute[grid_partial](xx, yy, a, n)