import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    products = a_vals * b_vals
    
    tl.store(dot_ptr + offsets, products, mask=mask)

def s352_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    dot_temp = torch.zeros_like(a)
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s352_kernel[grid](a, b, dot_temp, N, BLOCK_SIZE)
    
    dot = torch.sum(dot_temp)
    
    return dot.item()