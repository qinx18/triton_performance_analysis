import triton
import triton.language as tl
import torch

@triton.jit
def vpvts_kernel(a_ptr, b_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    result = a_vals + b_vals * s
    tl.store(a_ptr + indices, result, mask=mask)

def vpvts_triton(a, b, s):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpvts_kernel[grid](a, b, s, n_elements, BLOCK_SIZE)