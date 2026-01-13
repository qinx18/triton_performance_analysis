import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on conditional increments
    # We'll process one iteration at a time to maintain the sequential nature
    
    # Get the current iteration we're processing
    pid = tl.program_id(0)
    
    if pid >= n_half:
        return
    
    # Initialize j based on how many positive c values we've seen so far
    j = pid  # Base increment for each iteration
    
    # Count additional increments from previous iterations where c[k] > 0
    for k in range(pid):
        c_val = tl.load(c_ptr + k)
        if c_val > 0.0:
            j += 1
    
    # Process current iteration
    i = pid
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    e_val = tl.load(e_ptr + i)
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    result1 = b_val + d_val * e_val
    tl.store(a_ptr + j, result1)
    
    # Conditional assignment
    if c_val > 0.0:
        j += 1
        result2 = c_val + d_val * e_val
        tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    # Launch kernel with one thread per iteration to maintain sequential dependencies
    BLOCK_SIZE = 1
    grid = (n_half,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a