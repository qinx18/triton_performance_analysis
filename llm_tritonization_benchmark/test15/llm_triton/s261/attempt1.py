import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # c[i] depends on c[i-1] from previous iteration
    
    pid = tl.program_id(0)
    if pid > 0:
        return  # Only use first thread block
    
    # Process sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load scalar values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # t = a[i] + b[i]
        t = a_val + b_val
        
        # a[i] = t + c[i-1]
        tl.store(a_ptr + i, t + c_prev)
        
        # t = c[i] * d[i]
        t = c_val * d_val
        
        # c[i] = t
        tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread block since we need sequential execution
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )