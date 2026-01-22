import triton
import triton.language as tl
import torch

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (n - 1)
    
    # Load a[i+1] and b[i]
    a_vals = tl.load(a_ptr + offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s151_triton(a, b):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)