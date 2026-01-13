import triton
import triton.language as tl
import torch

@triton.jit
def s000_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    result = b_vals + 1.0
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s000_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s000_kernel[grid](a, b, n_elements, BLOCK_SIZE)