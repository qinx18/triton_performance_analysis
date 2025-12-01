import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a[i + k]
    read_indices = indices + k
    read_mask = mask & (read_indices < (n + k))
    a_read_vals = tl.load(a_ptr + read_indices, mask=read_mask)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_read_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s162_kernel[grid](a, b, c, n, k, BLOCK_SIZE)