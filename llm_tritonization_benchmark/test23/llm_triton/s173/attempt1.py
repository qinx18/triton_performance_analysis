import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(a_ptr, b_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < n
    
    # Load a[i] and b[i]
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+k]
    output_offsets = i_offsets + k
    tl.store(a_ptr + output_offsets, result, mask=mask)

def s173_triton(a, b, k):
    N = a.shape[0]
    n = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s173_kernel[grid](a, b, k, n, BLOCK_SIZE=BLOCK_SIZE)