import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    N: tl.constexpr,
):
    # This kernel must process sequentially due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Process all elements sequentially in a single thread
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Sequential processing from i=1 to N-1
    for i in range(1, N):
        # Load b[i-1]
        b_prev = tl.load(b_ptr + (i - 1))
        
        # Load c[i], d[i], e[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_val * e_val
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch with single program to ensure sequential execution
    grid = (1,)
    s323_kernel[grid](
        a, b, c, d, e,
        N=N,
    )
    
    return a, b