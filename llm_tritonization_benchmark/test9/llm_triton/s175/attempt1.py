import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Scale indices by inc to handle stride
    actual_indices = indices * inc
    
    mask = actual_indices < (n_elements - inc)
    
    # Load a[i + inc] and b[i]
    a_vals = tl.load(a_ptr + actual_indices + inc, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_vals + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    num_iterations = (n_elements - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, n_elements, inc,
        BLOCK_SIZE=BLOCK_SIZE
    )