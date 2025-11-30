import triton
import triton.language as tl
import torch

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Forward indices for memory access
    forward_offsets = offsets
    # Reverse indices for reading a
    reverse_offsets = n_elements - 1 - offsets
    reverse_mask = reverse_offsets >= 0
    combined_mask = mask & reverse_mask
    
    # Load data
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + forward_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + forward_offsets, mask=mask, other=0.0)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_reverse + b_vals * c_vals
    
    # Store results
    a_new = x - 1.0
    tl.store(a_ptr + forward_offsets, a_new, mask=mask)
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