import torch
import triton
import triton.language as tl

@triton.jit
def s151s_kernel(a_ptr, b_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute new values: a[i] = b[i] + inc
    a_vals = b_vals + inc
    
    # Store to a
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Call s151s with inc = 1
    s151s_kernel(a_ptr, b_ptr, n_elements, 1.0, BLOCK_SIZE)

def s151_triton(a, b):
    n_elements = a.numel()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return a