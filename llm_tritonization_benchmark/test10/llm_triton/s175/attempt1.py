import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices based on inc stride
    base_idx = block_start * inc
    indices = base_idx + offsets * inc
    
    # Mask for valid indices (i < LEN_1D-1)
    mask = (block_start + offsets < n_elements) & (indices < n_elements - 1)
    
    # Load data
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    a_inc_vals = tl.load(a_ptr + indices + inc, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + inc] + b[i]
    result = a_inc_vals + b_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s175_triton(a, b, inc):
    # Calculate number of elements to process
    n_elements = triton.cdiv(a.shape[0] - 1, inc)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a