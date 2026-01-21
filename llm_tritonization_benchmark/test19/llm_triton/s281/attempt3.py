import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    a_vals = tl.load(a_ptr + (n - 1 - offsets), mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute x = a[n-1-i] + b[i] * c[i]
    x_vals = a_vals + b_vals * c_vals
    
    # Store results
    tl.store(x_ptr + offsets, x_vals, mask=mask)
    tl.store(a_ptr + offsets, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + offsets, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create a copy of a for reading to handle crossing threshold
    a_copy = a.clone()
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s281_kernel[grid](a_copy, b, c, x, n, BLOCK_SIZE=BLOCK_SIZE)