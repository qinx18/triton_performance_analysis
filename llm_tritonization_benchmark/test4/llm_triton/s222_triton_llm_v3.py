import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially due to recurrence in e[i] = e[i-1] * e[i-1]
    for i in range(1, n_elements):
        # Load values
        bc_product = tl.load(b_ptr + i) * tl.load(c_ptr + i)
        
        # a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val += bc_product
        tl.store(a_ptr + i, a_val)
        
        # e[i] = e[i-1] * e[i-1] (recurrence - must be sequential)
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val -= bc_product
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Launch single thread due to recurrence dependency
    grid = (1,)
    s222_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=1024
    )
    
    return a, e