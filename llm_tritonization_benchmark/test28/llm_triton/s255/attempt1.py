import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load x and y values (broadcast to all threads in block)
    x = tl.load(b_ptr + n_elements - 1)
    y = tl.load(b_ptr + n_elements - 2)
    
    # Process each element in the block sequentially
    for i in range(BLOCK_SIZE):
        if block_start + i >= n_elements:
            break
            
        # Get current b value
        b_i = tl.load(b_ptr + block_start + i)
        
        # Compute a[i] = (b[i] + x + y) * 0.333
        result = (b_i + x + y) * 0.333
        
        # Store result
        tl.store(a_ptr + block_start + i, result)
        
        # Update x and y for next iteration
        y = x
        x = b_i

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    
    # Use a smaller block size since we need sequential processing
    BLOCK_SIZE = 128
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s255_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )