import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b = tl.load(b_ptr + offsets, mask=mask)
    condition = b > 0.0
    
    a_old = tl.load(a_ptr + offsets, mask=mask)
    a_new = tl.where(condition, b, a_old)
    
    tl.store(a_ptr + offsets, a_new, mask=mask)

def vif_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)