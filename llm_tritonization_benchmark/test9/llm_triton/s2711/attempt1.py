import triton
import triton.language as tl
import torch

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Check if b[i] != 0.0
    nonzero_mask = b_vals != 0.0
    update_mask = mask & nonzero_mask
    
    # Compute a[i] += b[i] * c[i] where b[i] != 0.0
    result = a_vals + b_vals * c_vals
    
    # Only store where update condition is met
    tl.store(a_ptr + idx, result, mask=update_mask)

def s2711_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)