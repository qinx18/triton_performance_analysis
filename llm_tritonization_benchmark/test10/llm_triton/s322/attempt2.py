import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a second-order linear recurrence
    # Each block processes sequentially since each element depends on previous elements
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    if block_start >= n_elements - 2:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially within each block
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i + 2  # Start from index 2
        
        if global_i >= n_elements:
            return
            
        # Load current values
        a_curr = tl.load(a_ptr + global_i)
        b_curr = tl.load(b_ptr + global_i)
        c_curr = tl.load(c_ptr + global_i)
        
        # Load previous values
        a_prev1 = tl.load(a_ptr + global_i - 1)
        a_prev2 = tl.load(a_ptr + global_i - 2)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        
        # Store result
        tl.store(a_ptr + global_i, result)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 32
    
    # Calculate grid size - we need to process n_elements - 2 elements (starting from index 2)
    grid_size = triton.cdiv(n_elements - 2, BLOCK_SIZE)
    
    # Launch kernel
    s322_kernel[(grid_size,)](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a