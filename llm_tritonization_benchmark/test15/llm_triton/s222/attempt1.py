import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # This kernel must process sequentially due to e[i] = e[i-1] * e[i-1]
    # Use only the first program to avoid race conditions
    if pid != 0:
        return
    
    # Process all elements sequentially
    for i in range(1, n_elements):
        # Load b[i] and c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        bc_product = b_val * c_val
        
        # Load a[i] and update: a[i] += b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val + bc_product
        tl.store(a_ptr + i, a_val)
        
        # Load e[i-1] and compute e[i] = e[i-1] * e[i-1]
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # Update a[i] again: a[i] -= b[i] * c[i]
        a_val = tl.load(a_ptr + i)
        a_val = a_val - bc_product
        tl.store(a_ptr + i, a_val)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Use minimal grid since only first program does work
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s222_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )