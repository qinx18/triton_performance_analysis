import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n - 1
    
    # Load b[i] for current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load a[j] = a[i+1] for current block
    j_offsets = offsets + 1
    a_j_vals = tl.load(a_ptr + j_offsets, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)