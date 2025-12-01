import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load data: a[i] and b[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute: a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k] where k = n (since n = LEN_1D/2 and k = LEN_1D/2)
    store_indices = indices + n
    tl.store(a_ptr + store_indices, result, mask=mask)

def s173_triton(a, b):
    n = a.shape[0] // 2  # LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s173_kernel[grid](a, b, n, BLOCK_SIZE)
    
    return a