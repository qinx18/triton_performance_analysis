import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the strictly sequential dependency
    # Only one thread processes all elements sequentially
    pid = tl.program_id(0)
    
    # Only thread 0 does the work to maintain sequential dependency
    if pid == 0:
        # Sequential loop - cannot be parallelized due to b[i] = ... b[i-1] dependency
        for i in range(1, n):
            # Load b[i-1], c[i], d[i], e[i]
            b_prev = tl.load(b_ptr + (i - 1))
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
    grid = (1,)  # Only one block since we need sequential processing
    
    s323_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )