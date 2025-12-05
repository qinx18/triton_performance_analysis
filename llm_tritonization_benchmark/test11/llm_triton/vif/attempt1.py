import triton
import triton.language as tl
import torch

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    condition = b_vals > 0.0
    
    combined_mask = mask & condition
    tl.store(a_ptr + idx, b_vals, mask=combined_mask)

def vif_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](a, b, n_elements, BLOCK_SIZE)