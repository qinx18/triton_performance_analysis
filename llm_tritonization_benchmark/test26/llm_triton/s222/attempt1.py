import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Since this has strict sequential dependency in e[i] = e[i-1] * e[i-1],
    # we must process sequentially. Use single thread.
    if pid != 0:
        return
    
    # Process all elements sequentially
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
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s222_kernel[grid](
        a, b, c, e, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, e