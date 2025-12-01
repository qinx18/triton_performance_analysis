import triton
import triton.language as tl
import torch

@triton.jit
def s1221_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute the range of elements this block will process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (starting from index 4)
    mask = (offsets >= 4) & (offsets < n_elements)
    
    # Load a[i] for current positions
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load b[i-4] for the dependency
    b_prev_offsets = offsets - 4
    b_prev_mask = (b_prev_offsets >= 0) & mask
    b_prev_vals = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask, other=0.0)
    
    # Compute b[i] = b[i-4] + a[i]
    result = b_prev_vals + a_vals
    
    # Store result back to b[i]
    tl.store(b_ptr + offsets, result, mask=mask)

def s1221_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1221_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return b