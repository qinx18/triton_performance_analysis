import torch
import triton
import triton.language as tl

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    condition = b_vals > 0.0
    
    # Only load from a where we need to potentially store
    combined_mask = mask & condition
    
    # Store b_vals to a where condition is true
    tl.store(a_ptr + indices, b_vals, mask=combined_mask)

def vif_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vif_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )