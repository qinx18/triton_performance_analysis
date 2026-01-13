import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block boundaries
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(0, BLOCK_SIZE):
        current_idx = block_start + i
        
        # Check bounds
        if current_idx >= n_elements:
            break
            
        # Load single elements
        a_val = tl.load(a_ptr + current_idx)
        c_val = tl.load(c_ptr + current_idx)
        b_val = tl.load(b_ptr + current_idx)
        d_val = tl.load(d_ptr + current_idx)
        
        # Load a[i+1] from copy
        next_idx = current_idx + 1
        if next_idx < (n_elements + 1):
            a_next_val = tl.load(a_copy_ptr + next_idx)
        else:
            a_next_val = 0.0
        
        # Perform computations
        new_a = a_val * c_val
        new_b = b_val + a_next_val * d_val
        
        # Store results
        tl.store(a_ptr + current_idx, new_a)
        tl.store(b_ptr + current_idx, new_b)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy
    a_copy = a.clone()
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s212_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )