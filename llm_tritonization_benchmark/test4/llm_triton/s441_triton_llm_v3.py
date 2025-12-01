import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # d[i] < 0: a[i] += b[i] * c[i]
    cond1 = d < 0.0
    result1 = a + b * c
    
    # d[i] == 0: a[i] += b[i] * b[i]
    cond2 = d == 0.0
    result2 = a + b * b
    
    # d[i] > 0: a[i] += c[i] * c[i]
    result3 = a + c * c
    
    # Apply conditions
    result = tl.where(cond1, result1, tl.where(cond2, result2, result3))
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)