import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    condition = b_vals > 0.0
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    result = tl.where(condition, b_vals, a_vals)
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def vif_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, n_elements, BLOCK_SIZE)