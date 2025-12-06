import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute condition: d[i] <= 0
    condition = d_vals <= 0.0
    
    # Compute both possible values
    val_l20 = b_vals * c_vals  # b[i] * c[i]
    val_l30 = b_vals * b_vals  # b[i] * b[i]
    
    # Select based on condition
    selected_val = tl.where(condition, val_l20, val_l30)
    
    # Update a[i]
    result = a_vals + selected_val
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )