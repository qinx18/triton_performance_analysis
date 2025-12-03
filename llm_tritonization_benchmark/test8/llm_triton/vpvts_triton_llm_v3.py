import torch
import triton
import triton.language as tl

@triton.jit
def vpvts_kernel(a_ptr, b_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    
    result = a_vals + b_vals * s
    
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def vpvts_triton(a, b, s):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpvts_kernel[grid](
        a, b, s, n_elements, BLOCK_SIZE
    )