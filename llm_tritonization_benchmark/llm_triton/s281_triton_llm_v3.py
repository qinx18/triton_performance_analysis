import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with reverse indexing for a
    reverse_offsets = n_elements - 1 - offsets
    reverse_mask = reverse_offsets >= 0
    
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=mask & reverse_mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x
    x = a_reverse + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s281_kernel[grid](
        a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )