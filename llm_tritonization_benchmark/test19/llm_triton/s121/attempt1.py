import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load a[j] = a[i+1] and b[i]
    a_j = tl.load(a_ptr + offsets + 1, mask=mask)
    b_i = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0] - 1  # Process LEN_1D-1 elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)