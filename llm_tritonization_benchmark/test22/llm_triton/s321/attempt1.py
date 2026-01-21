import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This is a first-order linear recurrence: a[i] += a[i-1] * b[i]
    # Must be computed sequentially, so we use a single thread
    pid = tl.program_id(0)
    
    # Only one thread does all the work due to sequential dependency
    if pid == 0:
        # Load first element
        prev_val = tl.load(a_ptr)
        
        # Process elements sequentially from 1 to n-1
        for i in range(1, n):
            # Load current values
            curr_a = tl.load(a_ptr + i)
            curr_b = tl.load(a_ptr + i)
            
            # Compute update: a[i] += a[i-1] * b[i]
            new_val = curr_a + prev_val * curr_b
            
            # Store result
            tl.store(a_ptr + i, new_val)
            
            # Update previous value for next iteration
            prev_val = new_val

def s321_triton(a, b):
    N = a.shape[0]
    
    # Launch single thread since this is a sequential recurrence
    grid = (1,)
    BLOCK_SIZE = 256
    
    s321_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )