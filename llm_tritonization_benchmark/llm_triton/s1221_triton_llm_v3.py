import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Start from index 4, so adjust offsets
    offsets = offsets + 4
    mask = offsets < n_elements
    
    # Load current elements
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Load b[i-4] elements
    b_prev_offsets = offsets - 4
    b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=mask)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_prev_vals + a_vals
    
    # Store result
    tl.store(b_ptr + offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    
    # We start from index 4, so we process n_elements - 4 elements
    elements_to_process = n_elements - 4
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s1221_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b