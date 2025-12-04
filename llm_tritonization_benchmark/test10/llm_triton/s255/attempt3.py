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
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Process each element sequentially within the block
    results = tl.zeros_like(b_vals)
    
    for idx in range(BLOCK_SIZE):
        i = block_start + idx
        if i >= n_elements:
            break
            
        # Calculate x and y for current iteration
        if i == 0:
            x = tl.load(b_ptr + n_elements - 1)
            y = tl.load(b_ptr + n_elements - 2)
        else:
            x = tl.load(b_ptr + i - 1)
            if i == 1:
                y = tl.load(b_ptr + n_elements - 1)
            else:
                y = tl.load(b_ptr + i - 2)
        
        # Load current b value
        b_i = tl.load(b_ptr + i)
        
        # Compute result
        result = (b_i + x + y) * 0.333
        
        # Store using select to maintain vectorization
        results = tl.where(tl.arange(0, BLOCK_SIZE) == idx, result, results)
    
    # Store results
    tl.store(a_ptr + offsets, results, mask=mask)

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )