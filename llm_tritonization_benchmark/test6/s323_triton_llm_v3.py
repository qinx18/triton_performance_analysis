import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially since this is a coupled recurrence
    for i in range(1, n_elements):
        # Load b[i-1] (scalar load)
        if i-1 < n_elements:
            b_prev = tl.load(b_ptr + (i-1))
        else:
            b_prev = 0.0
            
        # Load scalars for position i
        mask_i = i < n_elements
        if mask_i:
            c_i = tl.load(c_ptr + i)
            d_i = tl.load(d_ptr + i)
            e_i = tl.load(e_ptr + i)
            
            # Compute a[i] = b[i-1] + c[i] * d[i]
            a_i = b_prev + c_i * d_i
            tl.store(a_ptr + i, a_i)
            
            # Compute b[i] = a[i] + c[i] * e[i]
            b_i = a_i + c_i * e_i
            tl.store(b_ptr + i, b_i)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch single thread block since this is sequential
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )