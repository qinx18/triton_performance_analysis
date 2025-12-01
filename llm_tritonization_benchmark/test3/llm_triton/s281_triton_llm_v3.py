import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Forward offsets for b and c
    forward_offsets = offsets
    # Reverse offsets for a (reading)
    reverse_offsets = n_elements - 1 - offsets
    
    # Load data
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + forward_offsets, mask=mask)
    c_vals = tl.load(c_ptr + forward_offsets, mask=mask)
    
    # Compute x
    x = a_reverse + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + forward_offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + forward_offsets, x, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s281_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )