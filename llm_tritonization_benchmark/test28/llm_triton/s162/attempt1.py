import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy for anti-dependency safety
    a_copy_vals = tl.load(a_copy_ptr + indices + k, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Compute: a[i] = a[i + k] + b[i] * c[i]
    result = a_copy_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    N = a.shape[0]
    n = N - 1  # Loop goes from 0 to LEN_1D-2 (< LEN_1D-1)
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_copy, b, c, k, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )