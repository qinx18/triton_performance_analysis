import triton
import triton.language as tl
import torch

@triton.jit
def s443_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute conditional: if d[i] <= 0, use b*c, else use b*b
    condition = d_vals <= 0.0
    result1 = b_vals * c_vals  # L20 path
    result2 = b_vals * b_vals  # L30 path
    
    # Select result based on condition
    update = tl.where(condition, result1, result2)
    
    # Update a[i]
    new_a = a_vals + update
    
    tl.store(a_ptr + idx, new_a, mask=mask)

def s443_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s443_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )