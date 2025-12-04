import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n_elements
    
    # Initialize s as a scalar
    s = 0.0
    
    # Sequential loop over blocks
    for i in range(n_elements):
        # Load current values for this iteration
        if i < n_elements:
            e_val = tl.load(e_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + i, a_val)
            
            # s = b[i] + c[i]
            s = b_val + c_val
            
            # b[i] = a[i] + d[i]
            b_new_val = a_val + d_val
            tl.store(b_ptr + i, b_new_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Use single thread since computation is inherently serial
    BLOCK_SIZE = 1
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )