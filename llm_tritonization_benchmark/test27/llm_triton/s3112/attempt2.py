import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n_elements):
    # This is a sequential cumulative sum - must be processed by single thread
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first program processes all data
        running_sum = 0.0
        
        # Process all elements sequentially
        for i in range(n_elements):
            # Load single element
            val = tl.load(a_ptr + i)
            running_sum += val
            # Store cumulative sum
            tl.store(b_ptr + i, running_sum)

def s3112_triton(a, b):
    n_elements = a.shape[0]
    
    # Launch with single thread since this is inherently sequential
    s3112_kernel[(1,)](
        a, b, n_elements
    )
    
    # Return the final sum (last element of b)
    return b[-1].item()