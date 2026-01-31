import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    condition = b_vals > 0.0
    
    # Only load and compute where condition is true and within bounds
    compute_mask = mask & condition
    
    a_vals = tl.load(a_ptr + indices, mask=compute_mask)
    c_vals = tl.load(c_ptr + indices, mask=compute_mask)
    
    # Compute and store the result
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=compute_mask)

def s271_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )