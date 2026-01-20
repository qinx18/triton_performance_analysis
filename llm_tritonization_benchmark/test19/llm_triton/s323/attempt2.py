import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire array sequentially since there's a loop-carried dependency
    # Only one thread block should run this kernel
    pid = tl.program_id(0)
    
    if pid == 0:  # Only the first block processes everything
        for i in range(1, n_elements):
            # Load b[i-1], c[i], d[i], e[i]
            b_prev = tl.load(b_ptr + (i - 1))
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # Compute a[i] = b[i-1] + c[i] * d[i]
            a_val = b_prev + c_val * d_val
            
            # Store a[i]
            tl.store(a_ptr + i, a_val)
            
            # Compute b[i] = a[i] + c[i] * e[i]
            b_val = a_val + c_val * e_val
            
            # Store b[i]
            tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use only one thread block since computation is strictly sequential
    grid = (1,)
    
    # Launch kernel
    s323_kernel[grid](
        a, b, c, d, e, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )