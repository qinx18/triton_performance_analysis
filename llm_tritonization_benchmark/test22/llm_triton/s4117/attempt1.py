import triton
import triton.language as tl
import torch

@triton.jit
def s4117_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    c_indices = offsets // 2
    c_vals = tl.load(c_ptr + c_indices, mask=mask)
    
    result = b_vals + c_vals * d_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s4117_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4117_kernel[grid](a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE)