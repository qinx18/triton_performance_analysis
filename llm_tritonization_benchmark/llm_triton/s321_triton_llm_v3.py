import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    prev_offsets = offsets - 1
    prev_mask = prev_offsets >= 0
    
    a_prev = tl.load(a_copy_ptr + prev_offsets, mask=prev_mask & mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    a_curr = tl.load(a_copy_ptr + offsets, mask=mask, other=0.0)
    
    result = a_curr + a_prev * b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = ((n_elements - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    a_copy = a.clone()
    
    s321_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )