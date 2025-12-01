import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Condition: if (i+1 < mid)
    condition_mask = (offsets + 1) < mid
    
    # Compute: a[i] += b[i] * c[i] if condition, else a[i] += b[i] * d[i]
    bc_product = b_vals * c_vals
    bd_product = b_vals * d_vals
    
    # Select based on condition
    product = tl.where(condition_mask, bc_product, bd_product)
    result = a_vals + product
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s276_triton(a, b, c, d):
    n_elements = a.numel()
    mid = n_elements // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )