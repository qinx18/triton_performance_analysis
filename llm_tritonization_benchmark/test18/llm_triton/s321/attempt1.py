import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This is a first-order linear recurrence: a[i] += a[i-1] * b[i]
    # Must be computed sequentially as each element depends on the previous
    
    # Process elements sequentially in blocks
    for block_start in range(0, N - 1, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, N - 1)
        
        # Process each element in this block sequentially
        for offset in range(block_start, block_end):
            i = offset + 1  # Start from index 1
            
            # Load a[i-1], a[i], and b[i]
            a_prev = tl.load(a_ptr + (i - 1))
            a_curr = tl.load(a_ptr + i)
            b_curr = tl.load(b_ptr + i)
            
            # Compute a[i] += a[i-1] * b[i]
            result = a_curr + a_prev * b_curr
            
            # Store result
            tl.store(a_ptr + i, result)

def s321_triton(a, b):
    N = a.shape[0]
    
    if N <= 1:
        return
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program instance since this is sequential
    s321_kernel[(1,)](
        a, b, N, BLOCK_SIZE
    )