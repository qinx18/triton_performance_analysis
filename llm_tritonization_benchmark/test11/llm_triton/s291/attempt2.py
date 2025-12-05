import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current b values
    b_current = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate previous indices with wrap-around
    prev_offsets = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    
    # Load previous b values
    b_prev = tl.load(b_ptr + prev_offsets, mask=mask, other=0.0)
    
    # Compute result
    result = (b_current + b_prev) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a