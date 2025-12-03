import torch
import triton
import triton.language as tl

@triton.jit
def s4121_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    
    # Compute f(b[i], c[i]) = b[i] * c[i] and add to a[i]
    result = a_vals + (b_vals * c_vals)
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s4121_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4121_kernel[grid](
        a, b, c, n_elements, BLOCK_SIZE
    )