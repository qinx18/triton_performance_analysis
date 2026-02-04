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
    
    # Each block processes BLOCK_SIZE elements sequentially within its range
    for offset in range(BLOCK_SIZE):
        i = block_start + offset
        if i >= N:
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
    
    BLOCK_SIZE = 64
    # Calculate grid size for elements starting from index 2
    num_elements = N - 2
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    s322_kernel[grid](a, b, c, N, BLOCK_SIZE)
    
    return a