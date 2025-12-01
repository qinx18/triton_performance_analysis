import triton
import triton.language as tl
import torch

@triton.jit
def s171_kernel(
    a_ptr, b_ptr,
    n_elements, inc,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_offsets = offsets * inc
    a_mask = a_offsets < (n_elements * inc)
    combined_mask = mask & a_mask
    
    a_vals = tl.load(a_ptr + a_offsets, mask=combined_mask)
    result = a_vals + b_vals
    tl.store(a_ptr + a_offsets, result, mask=combined_mask)

def s171_triton(a, b, inc):
    n_elements = a.shape[0] // inc if inc > 1 else a.shape[0]
    n_elements = min(n_elements, b.shape[0])
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s171_kernel[grid](
        a, b,
        n_elements, inc,
        BLOCK_SIZE=BLOCK_SIZE,
    )