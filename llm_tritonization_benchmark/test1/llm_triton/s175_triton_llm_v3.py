import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear indices to strided indices (i = 0, inc, 2*inc, ...)
    actual_indices = offsets * inc
    mask = actual_indices < (n_elements - 1)
    
    # Load a[i + inc] and b[i]
    a_read_indices = actual_indices + inc
    a_read_mask = mask & (a_read_indices < n_elements)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    
    # Calculate number of iterations in the strided loop
    num_iterations = (n_elements - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b,
        n_elements, inc,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a