import triton
import triton.language as tl
import torch

@triton.jit
def s276_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, mid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Condition: (i+1) < mid, which is equivalent to i < (mid-1)
    condition = indices < (mid - 1)
    
    # Compute products
    bc_product = b_vals * c_vals
    bd_product = b_vals * d_vals
    
    # Select product based on condition
    selected_product = tl.where(condition, bc_product, bd_product)
    
    # Update a
    new_a = a_vals + selected_product
    
    # Store result
    tl.store(a_ptr + indices, new_a, mask=mask)

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