import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to loop-carried dependency on scalar s
    # Only one thread block processes the entire array
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    s = 0.0
    
    # Process each element sequentially
    for i in range(n_elements):
        # a[i] = s * e[i]
        e_val = tl.load(e_ptr + i)
        a_val = s * e_val
        tl.store(a_ptr + i, a_val)
        
        # s = b[i] + c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s = b_val + c_val
        
        # b[i] = a[i] + d[i]
        d_val = tl.load(d_ptr + i)
        new_b_val = a_val + d_val
        tl.store(b_ptr + i, new_b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Launch with only 1 thread block since we need sequential processing
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b