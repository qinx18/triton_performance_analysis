import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from the previous iteration
    pid = tl.program_id(0)
    
    # Only run on the first thread to maintain sequential dependency
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(1, n):
        # Load b[i-1], c[i], d[i], e[i]
        b_prev = tl.load(b_ptr + i - 1)
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
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use only one program since we need sequential execution
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )