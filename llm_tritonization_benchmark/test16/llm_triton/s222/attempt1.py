import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, N):
    # This kernel must process sequentially due to e[i] = e[i-1] * e[i-1] dependency
    # Use single thread to handle the entire computation
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(1, N):
        # Load values for a[i] += b[i] * c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        bc_product = b_val * c_val
        
        # a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val + bc_product
        tl.store(a_ptr + i, a_val)
        
        # e[i] = e[i - 1] * e[i - 1]
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val - bc_product
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    N = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s222_kernel[grid](
        a, b, c, e, N
    )
    
    return a, e