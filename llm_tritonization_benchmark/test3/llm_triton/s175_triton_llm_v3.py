import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert to actual indices based on stride
    actual_indices = offsets * inc
    
    # Mask for valid indices
    mask = actual_indices < n_elements
    
    # Load a[i + inc] and b[i]
    a_read_indices = actual_indices + inc
    a_read_mask = mask & (a_read_indices < (n_elements + inc))
    
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store back to a[i]
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0] - 1
    num_iterations = (n_elements + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )