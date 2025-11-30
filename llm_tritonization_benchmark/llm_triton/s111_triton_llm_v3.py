import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements (only odd indices from 1 to n_elements-1)
    mask = (offsets < n_elements) & ((offsets % 2) == 1) & (offsets >= 1)
    
    # Load a[i-1] and b[i] for valid odd indices
    a_prev_offsets = offsets - 1
    a_prev_mask = (a_prev_offsets >= 0) & (a_prev_offsets < n_elements) & mask
    b_mask = mask
    
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=b_mask, other=0.0)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s111_kernel[(grid_size,)](
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a