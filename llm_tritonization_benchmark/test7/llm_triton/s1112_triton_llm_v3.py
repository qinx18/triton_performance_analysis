import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Reverse the indices for backward iteration
    indices = (n_elements - 1) - (block_start + offsets)
    mask = (block_start + offsets) < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = b[i] + 1.0
    a_vals = b_vals + 1.0
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s1112_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )