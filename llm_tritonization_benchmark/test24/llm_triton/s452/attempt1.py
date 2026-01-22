import triton
import triton.language as tl
import torch

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    indices = offsets + 1
    result = b_vals + c_vals * indices.to(tl.float32)
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s452_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s452_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)