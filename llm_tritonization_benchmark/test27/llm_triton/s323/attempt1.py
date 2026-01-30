import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from index 1 to n_elements-1
    for i in range(1, n_elements):
        # Load b[i-1] (from previous iteration)
        b_prev = tl.load(b_ptr + (i - 1))
        
        # Load current elements
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_i = b_prev + c_i * d_i
        
        # Store a[i]
        tl.store(a_ptr + i, a_i)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_i = a_i + c_i * e_i
        
        # Store b[i]
        tl.store(b_ptr + i, b_i)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Use single block since we need sequential processing
    BLOCK_SIZE = 1
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b