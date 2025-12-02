import torch
import triton
import triton.language as tl

@triton.jit
def s151s_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Compute: a[i] = (a[i] + b[0]) * 0.5
    b0_val = tl.load(b_ptr)  # Load b[0]
    result = (a_vals + b0_val) * 0.5
    
    # Store back to a
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    s151s_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, BLOCK_SIZE)