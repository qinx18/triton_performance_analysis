import triton
import triton.language as tl
import torch

@triton.jit
def s174_kernel(a_ptr, b_ptr, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < M
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+M]
    output_indices = indices + M
    tl.store(a_ptr + output_indices, result, mask=mask)

def s174_triton(a, b, M):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    s174_kernel[grid](a, b, M, BLOCK_SIZE=BLOCK_SIZE)
    
    return a