import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    for i in range(1, n):
        # Load b[i] and c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        bc_product = b_val * c_val
        
        # a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        new_a_val = a_val + bc_product
        tl.store(a_ptr + i, new_a_val)
        
        # e[i] = e[i-1] * e[i-1]
        e_prev = tl.load(e_ptr + (i - 1))
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # a[i] -= b[i] * c[i] (reload a[i] to ensure consistency)
        a_current = tl.load(a_ptr + i)
        final_a_val = a_current - bc_product
        tl.store(a_ptr + i, final_a_val)

def s222_triton(a, b, c, e):
    n = a.shape[0]
    grid = (1,)
    s222_kernel[grid](a, b, c, e, n)
    return a, e