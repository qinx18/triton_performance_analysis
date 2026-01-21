import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a stream compaction operation that must be done sequentially
    # We'll process the entire array in one thread block
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Initialize counter
    j = -1
    
    # Process elements sequentially
    for i in range(n_elements):
        # Load b[i]
        b_val = tl.load(b_ptr + i)
        
        # Check condition
        if b_val > 0.0:
            j += 1
            # Store b[i] to a[j]
            tl.store(a_ptr + j, b_val)

def s341_triton(a, b):
    N = a.shape[0]
    
    # Stream compaction requires sequential processing
    # Use single thread block approach
    BLOCK_SIZE = 1
    grid = (1,)
    
    s341_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )