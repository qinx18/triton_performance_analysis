import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential recurrence: e[i] = e[i-1] * e[i-1] prevents vectorization
    # Process elements sequentially starting from index 1
    for i in range(1, n_elements):
        # Load scalar values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        bc_prod = b_val * c_val
        
        # Update a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val + bc_prod
        tl.store(a_ptr + i, a_val)
        
        # Sequential recurrence: e[i] = e[i-1] * e[i-1]
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # Update a[i] -= b[i] * c[i] (net effect cancels first operation)
        a_val = tl.load(a_ptr + i)
        a_val = a_val - bc_prod
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s222_kernel[grid](
        a, b, c, e, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )