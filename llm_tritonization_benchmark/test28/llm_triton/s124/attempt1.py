import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize output position counter for this block
    output_pos = block_start
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        element_idx = block_start + i
        
        # Check bounds
        if element_idx >= n_elements:
            break
            
        # Load single elements
        b_val = tl.load(b_ptr + element_idx)
        c_val = tl.load(c_ptr + element_idx)
        d_val = tl.load(d_ptr + element_idx)
        e_val = tl.load(e_ptr + element_idx)
        
        # Compute result based on condition
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            result = c_val + d_val * e_val
            
        # Store result at current output position
        tl.store(a_ptr + output_pos, result)
        output_pos += 1

def s124_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s124_kernel[(grid_size,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a