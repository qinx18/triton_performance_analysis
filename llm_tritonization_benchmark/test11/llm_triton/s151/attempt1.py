import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid elements (excluding last element since we go up to n_elements-1)
    mask = idx < (n_elements - 1)
    
    # Load a[i+1] (shifted by m=1) and b[i]
    a_shifted = tl.load(a_ptr + idx + 1, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_shifted + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.size(0)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, BLOCK_SIZE)