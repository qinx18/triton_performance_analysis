import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional computation: if (i+1 < mid)
    condition = (offsets + 1) < mid
    
    # Compute based on condition
    result = tl.where(condition, a + b * c, a + b * d)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s276_triton(a, b, c, d):
    n_elements = a.numel()
    mid = n_elements // 2
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s276_kernel[grid](a, b, c, d, n_elements, mid, BLOCK_SIZE=BLOCK_SIZE)
    
    return a