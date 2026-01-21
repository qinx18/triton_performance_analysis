import triton
import triton.language as tl
import torch

@triton.jit
def vpvts_kernel(a_ptr, b_ptr, s, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals * s
    
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvts_triton(a, b, s):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vpvts_kernel[grid](a, b, s, N, BLOCK_SIZE)