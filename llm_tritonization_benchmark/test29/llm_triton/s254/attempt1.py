import triton
import triton.language as tl
import torch

@triton.jit
def s254_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute x values for each position
    # For i=0: x = b[n_elements-1]
    # For i>0: x = b[i-1]
    prev_indices = tl.where(indices == 0, n_elements - 1, indices - 1)
    x_vals = tl.load(b_ptr + prev_indices, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + indices, result, mask=mask)

def s254_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s254_kernel[grid](a, b, n_elements, BLOCK_SIZE)