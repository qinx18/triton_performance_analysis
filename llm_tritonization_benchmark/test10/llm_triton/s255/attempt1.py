import torch
import triton
import triton.language as tl

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize x and y from the end of array b
    x = tl.load(b_ptr + n_elements - 1)
    y = tl.load(b_ptr + n_elements - 2)
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        current_idx = block_start + i
        
        if current_idx < n_elements:
            # Load b[i]
            b_val = tl.load(b_ptr + current_idx)
            
            # Compute a[i] = (b[i] + x + y) * 0.333
            result = (b_val + x + y) * 0.333
            
            # Store result
            tl.store(a_ptr + current_idx, result)
            
            # Update carry variables
            y = x
            x = b_val

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