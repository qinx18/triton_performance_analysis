import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] where i ranges from 0 to LEN_1D-1 with stride inc
    actual_indices = indices * inc
    b_vals = tl.load(b_ptr + actual_indices, mask=mask, other=0.0)
    a_inc_vals = tl.load(a_ptr + actual_indices + inc, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_inc_vals + b_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s175_triton(a, b, inc):
    # Number of iterations: i goes from 0 to LEN_1D-1 with step inc
    # So we have (LEN_1D-1 + inc - 1) // inc iterations
    n_elements = (a.shape[0] - 1 + inc - 1) // inc
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a