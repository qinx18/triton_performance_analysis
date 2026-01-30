import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel must be sequential due to loop-carried dependency c[i] reads c[i-1]
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one thread processes all elements sequentially
        for i in range(1, N):
            # Load values for current iteration
            a_i = tl.load(a_ptr + i)
            b_i = tl.load(b_ptr + i)
            c_i_minus_1 = tl.load(c_ptr + i - 1)
            c_i = tl.load(c_ptr + i)
            d_i = tl.load(d_ptr + i)
            
            # First computation: t = a[i] + b[i]
            t = a_i + b_i
            
            # Store a[i] = t + c[i-1]
            a_result = t + c_i_minus_1
            tl.store(a_ptr + i, a_result)
            
            # Second computation: t = c[i] * d[i]
            t = c_i * d_i
            
            # Store c[i] = t
            tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 128
    
    # Calculate grid size - only need one block since processing is sequential
    grid = (1,)
    
    # Launch kernel
    s261_kernel[grid](
        a, b, c, d,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c