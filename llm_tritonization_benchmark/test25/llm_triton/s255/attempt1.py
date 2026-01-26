import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load x and y values
    x = tl.load(b_ptr + (n_elements - 1))
    y = tl.load(b_ptr + (n_elements - 2))
    
    # Process elements in this block
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x + y) * 0.333
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s255_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a