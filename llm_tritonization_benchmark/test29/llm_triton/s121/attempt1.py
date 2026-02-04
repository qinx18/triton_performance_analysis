import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N - 1
    
    # Load a[i+1] and b[i]
    a_vals = tl.load(a_ptr + offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_vals + b_vals
    
    # Store result to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Copy a for read to avoid WAR dependency
    a_copy = a.clone()
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](a, a_copy, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return a