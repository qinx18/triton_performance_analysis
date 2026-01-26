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
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # For each element, compute x and y values based on loop iteration
    # x starts as b[n-1], y starts as b[n-2]
    # After i iterations: y = b[i-2] (or original values), x = b[i-1] (or original values)
    
    x_vals = tl.where(current_offsets == 0, 
                      tl.load(b_ptr + (n_elements - 1)),
                      tl.where(current_offsets == 1,
                               tl.load(b_ptr + 0),
                               tl.load(b_ptr + current_offsets - 1)))
    
    y_vals = tl.where(current_offsets == 0,
                      tl.load(b_ptr + (n_elements - 2)),
                      tl.where(current_offsets == 1,
                               tl.load(b_ptr + (n_elements - 1)),
                               tl.load(b_ptr + current_offsets - 2)))
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x_vals + y_vals) * 0.333
    
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