import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles second order linear recurrence sequentially
    # Each thread block processes BLOCK_SIZE consecutive elements
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block
    for local_idx in range(BLOCK_SIZE):
        global_idx = block_start + local_idx
        
        if global_idx >= 2 and global_idx < n_elements:
            # Load current values
            a_val = tl.load(a_ptr + global_idx)
            b_val = tl.load(b_ptr + global_idx)
            c_val = tl.load(c_ptr + global_idx)
            
            # Load dependencies
            a_prev1 = tl.load(a_ptr + global_idx - 1)
            a_prev2 = tl.load(a_ptr + global_idx - 2)
            
            # Compute recurrence
            result = a_val + a_prev1 * b_val + a_prev2 * c_val
            
            # Store result
            tl.store(a_ptr + global_idx, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Use small block size to maintain sequential dependencies
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel with sequential execution
    s322_kernel[(grid_size,)](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a