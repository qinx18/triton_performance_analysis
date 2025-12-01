import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(
    a_ptr, b_ptr,
    inc,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert offsets to actual indices based on stride
    indices = offsets * inc
    mask = indices < (n_elements - 1)
    
    # Load data with masking
    b_vals = tl.load(b_ptr + indices, mask=mask)
    a_inc_vals = tl.load(a_ptr + indices + inc, mask=mask)
    
    # Compute result
    result = a_inc_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    # Calculate number of iterations based on stride
    n_iterations = (a.shape[0] - 1 + inc - 1) // inc
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_iterations, meta['BLOCK_SIZE']),)
    
    s175_kernel[grid](
        a, b,
        inc,
        a.shape[0],
        BLOCK_SIZE=256,
    )
    
    return a