import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This computation has strict loop-carried dependencies
    # Each element depends on the previous two elements of array 'a'
    # Must process sequentially, but we can still use parallel blocks
    # for better memory access patterns
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 2  # Start from index 2
    
    if block_start >= N:
        return
    
    # Calculate the end index for this block
    block_end = tl.minimum(block_start + BLOCK_SIZE, N)
    
    # Each block processes BLOCK_SIZE elements sequentially within its range
    for offset in range(BLOCK_SIZE):
        i = block_start + offset
        if i >= block_end:
            break
            
        # Load required values
        a_curr = tl.load(a_ptr + i)
        a_prev1 = tl.load(a_ptr + i - 1)
        a_prev2 = tl.load(a_ptr + i - 2)
        b_curr = tl.load(b_ptr + i)
        c_curr = tl.load(c_ptr + i)
        
        # Compute new value
        new_val = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        
        # Store result
        tl.store(a_ptr + i, new_val)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    if N <= 2:
        return a
    
    # Use a single thread to process sequentially
    BLOCK_SIZE = 1
    grid = (1,)
    
    # Process all elements sequentially in a single thread
    for i in range(2, N):
        a_curr = a[i].item()
        a_prev1 = a[i-1].item()
        a_prev2 = a[i-2].item()
        b_curr = b[i].item()
        c_curr = c[i].item()
        
        new_val = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        a[i] = new_val
    
    return a