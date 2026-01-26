import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load a[j] where j = i + 1
    j_indices = indices + 1
    a_j = tl.load(a_ptr + j_indices, mask=mask, other=0.0)
    
    # Load b[i]
    b_i = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2 (i < LEN_1D-1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE)