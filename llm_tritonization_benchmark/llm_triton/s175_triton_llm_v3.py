import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear offsets to actual indices based on increment
    indices = offsets * inc
    
    # Mask for valid indices
    mask = indices < (n_elements - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask)
    a_offset_vals = tl.load(a_ptr + indices + inc, mask=mask)
    
    # Compute result
    result = a_offset_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0]
    
    # Calculate number of iterations based on increment
    num_iterations = (n_elements - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a