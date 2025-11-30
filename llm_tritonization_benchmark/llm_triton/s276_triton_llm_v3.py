import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    mid,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Create condition mask for i+1 < mid
    condition_mask = (offsets + 1) < mid
    
    # Load c and d values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load current a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute based on condition
    # If i+1 < mid: a[i] += b[i] * c[i]
    # Else: a[i] += b[i] * d[i]
    bc_product = b_vals * c_vals
    bd_product = b_vals * d_vals
    
    # Select appropriate product based on condition
    selected_product = tl.where(condition_mask, bc_product, bd_product)
    
    # Update a values
    new_a_vals = a_vals + selected_product
    
    # Store result
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)

def s276_triton(a, b, c, d):
    n_elements = a.numel()
    mid = n_elements // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s276_kernel[grid](
        a, b, c, d,
        n_elements,
        mid,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a