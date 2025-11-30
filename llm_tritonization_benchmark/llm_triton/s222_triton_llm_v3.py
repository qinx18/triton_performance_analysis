import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the recurrence in e[i] = e[i-1] * e[i-1]
    # Must be executed sequentially due to the dependency
    for i in range(1, n_elements):
        if i < n_elements:
            # Load values
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            bc_product = b_val * c_val
            
            # First operation: a[i] += b[i] * c[i]
            a_val = tl.load(a_ptr + i)
            a_val = a_val + bc_product
            
            # Recurrence: e[i] = e[i-1] * e[i-1]
            e_prev = tl.load(e_ptr + i - 1)
            e_val = e_prev * e_prev
            tl.store(e_ptr + i, e_val)
            
            # Third operation: a[i] -= b[i] * c[i]
            a_val = a_val - bc_product
            tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Launch single-threaded kernel due to recurrence
    s222_kernel[(1,)](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=1
    )
    
    return a