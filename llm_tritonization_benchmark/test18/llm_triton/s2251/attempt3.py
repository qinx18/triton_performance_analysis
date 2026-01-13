import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to sequential dependency on scalar s
    # We need to process all elements sequentially in a single thread block
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially one by one
    s = 0.0
    
    for i in range(n_elements):
        # Load individual elements
        e_val = tl.load(e_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # a[i] = s * e[i]
        a_val = s * e_val
        
        # s = b[i] + c[i]
        s = b_val + c_val
        
        # b[i] = a[i] + d[i]
        b_new_val = a_val + d_val
        
        # Store results
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_new_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )