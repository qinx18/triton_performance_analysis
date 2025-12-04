import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all b values for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Initialize x and y from the end of array b
    x = tl.load(b_ptr + n_elements - 1)
    y = tl.load(b_ptr + n_elements - 2)
    
    # Compute results
    results = (b_vals + x + y) * 0.333
    
    # Store results
    tl.store(a_ptr + offsets, results, mask=mask)

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )