import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    condition = b_vals > 0.0
    
    combined_mask = mask & condition
    tl.store(a_ptr + offsets, b_vals, mask=combined_mask)

def vif_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)