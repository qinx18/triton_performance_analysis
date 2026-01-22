import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel computes s += 2.0 for each i, then a[i] = s * b[i]
    # Since s = 2.0 * (i + 1), we can compute this directly
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute s for each element: s = 2.0 * (i + 1)
    s_vals = 2.0 * (offsets.to(tl.float32) + 1.0)
    
    # Compute a[i] = s * b[i]
    a_vals = s_vals * b_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s453_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)