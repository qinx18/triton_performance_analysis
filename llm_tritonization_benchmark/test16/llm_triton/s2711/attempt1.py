import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Check condition: b[i] != 0.0
    condition = b_vals != 0.0
    
    # Compute: a[i] += b[i] * c[i] where condition is true
    update = b_vals * c_vals
    result = tl.where(condition, a_vals + update, a_vals)
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s2711_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )