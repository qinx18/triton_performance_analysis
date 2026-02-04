import triton
import triton.language as tl
import torch

@triton.jit
def vpv_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_vals + b_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def vpv_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vpv_kernel[grid](a, b, N, BLOCK_SIZE)