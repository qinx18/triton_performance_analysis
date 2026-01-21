import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    condition = b_vals > 0.0
    
    combined_mask = mask & condition
    
    a_vals = tl.load(a_ptr + indices, mask=combined_mask)
    c_vals = tl.load(c_ptr + indices, mask=combined_mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=combined_mask)

def s271_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s271_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)