import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load current b values
    b_current = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute previous indices (wrap around for i=0)
    prev_offsets = tl.where(offsets == 0, n_elements - 1, offsets - 1)
    
    # Load previous b values
    b_prev = tl.load(b_ptr + prev_offsets, mask=mask)
    
    # Compute result: (b[i] + b[im1]) * 0.5
    result = (b_current + b_prev) * 0.5
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s291_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )