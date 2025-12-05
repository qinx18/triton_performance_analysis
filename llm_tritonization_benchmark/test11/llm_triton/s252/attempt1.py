import torch
import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a sequential scan operation
    # Each block processes a contiguous chunk of the array
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize t to 0 for this block
    t = 0.0
    
    # Process elements sequentially within this block
    for i in range(BLOCK_SIZE):
        current_idx = block_start + i
        
        if current_idx < n_elements:
            # Load single elements
            b_val = tl.load(b_ptr + current_idx)
            c_val = tl.load(c_ptr + current_idx)
            
            # Compute s = b[i] * c[i]
            s = b_val * c_val
            
            # Compute a[i] = s + t
            a_val = s + t
            
            # Store result
            tl.store(a_ptr + current_idx, a_val)
            
            # Update t = s for next iteration
            t = s

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Use small block size to maintain sequential dependency
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )