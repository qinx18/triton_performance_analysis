import torch
import triton
import triton.language as tl

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current block
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load previous values (shift by -1)
    prev_offsets = offsets - 1
    prev_mask = (offsets > 0) & (offsets < n_elements)
    a_prev = tl.load(a_ptr + prev_offsets, mask=prev_mask)
    
    # Compute update: a[i] += a[i-1] * b[i]
    new_vals = tl.where(offsets > 0, a_vals + a_prev * b_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + offsets, new_vals, mask=mask)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s321_kernel[grid](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )