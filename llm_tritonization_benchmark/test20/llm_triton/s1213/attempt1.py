import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Sequential processing - each block handles a chunk sequentially
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Process sequentially within each block
    for offset in range(BLOCK_SIZE):
        i = block_start + offset + 1  # Start from 1
        
        if i < n - 1:  # Ensure i < LEN_1D - 1
            # Load values
            b_prev = tl.load(b_ptr + (i - 1))
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # First statement: a[i] = b[i-1] + c[i]
            a_val = b_prev + c_val
            tl.store(a_ptr + i, a_val)
            
            # Load a[i+1] for second statement
            a_next = tl.load(a_ptr + (i + 1))
            
            # Second statement: b[i] = a[i+1] * d[i]
            b_val = a_next * d_val
            tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for a array to handle WAR dependency
    a_copy = a.clone()
    
    # Calculate grid size for range [1, n-1)
    num_elements = n - 2
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    # Launch kernel with original a for writes and copy for reads
    s1213_kernel[grid](
        a, a_copy, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )