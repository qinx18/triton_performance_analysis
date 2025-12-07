import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Conditional computation: if (i+1 < mid)
    condition = (idx + 1) < mid
    
    # Select between b*c and b*d based on condition
    result = tl.where(condition, b_vals * c_vals, b_vals * d_vals)
    
    # Update a[i] += result
    new_a = a_vals + result
    
    # Store result
    tl.store(a_ptr + idx, new_a, mask=mask)

def s276_triton(a, b, c, d, mid):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a