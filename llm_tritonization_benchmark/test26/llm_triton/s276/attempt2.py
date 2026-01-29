import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Condition: if (i+1 < mid)
    condition = (idx + 1) < mid
    
    # Compute: a[i] += b[i] * c[i] or a[i] += b[i] * d[i]
    multiply_vals = tl.where(condition, b_vals * c_vals, b_vals * d_vals)
    result = a_vals + multiply_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s276_triton(a, b, c, d):
    n_elements = a.shape[0]
    mid = n_elements // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )