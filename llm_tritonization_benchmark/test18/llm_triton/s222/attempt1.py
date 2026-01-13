import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements):
    # This kernel must be strictly sequential due to e[i] = e[i-1] * e[i-1]
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially starting from index 1
    for i in range(1, n_elements):
        # Load b[i] and c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        bc_product = b_val * c_val
        
        # a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val + bc_product
        tl.store(a_ptr + i, a_val)
        
        # e[i] = e[i-1] * e[i-1]
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val - bc_product
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Launch with single block, single thread due to sequential dependency
    grid = (1,)
    s222_kernel[grid](
        a, b, c, e, n_elements
    )