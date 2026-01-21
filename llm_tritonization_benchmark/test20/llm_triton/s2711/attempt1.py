import triton
import triton.language as tl
import torch

@triton.jit
def s2711_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Check if b[i] != 0.0
    nonzero_mask = b_vals != 0.0
    combined_mask = mask & nonzero_mask
    
    # Compute a[i] += b[i] * c[i] only where b[i] != 0
    update_vals = a_vals + b_vals * c_vals
    result = tl.where(nonzero_mask, update_vals, a_vals)
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s2711_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2711_kernel[grid](a, b, c, N, BLOCK_SIZE)