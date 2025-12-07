import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(
    a_ptr, b_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Conditional assignment: if b[i] > 0, a[i] = b[i]
    condition = b_vals > 0.0
    
    # Load current a values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Update a values where condition is true
    result = tl.where(condition, b_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def vif_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )