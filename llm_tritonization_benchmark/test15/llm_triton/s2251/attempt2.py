import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process elements sequentially due to dependencies
    # Each kernel instance processes one element at a time
    idx = tl.program_id(0)
    
    if idx >= n_elements:
        return
    
    # Get previous s value
    if idx == 0:
        s = 0.0
    else:
        # s = b[i-1] + c[i-1]
        b_prev = tl.load(b_ptr + (idx - 1))
        c_prev = tl.load(c_ptr + (idx - 1))
        s = b_prev + c_prev
    
    # Load current values
    e_val = tl.load(e_ptr + idx)
    d_val = tl.load(d_ptr + idx)
    
    # Compute a[i] = s * e[i]
    a_val = s * e_val
    tl.store(a_ptr + idx, a_val)
    
    # Compute b[i] = a[i] + d[i]
    b_new = a_val + d_val
    tl.store(b_ptr + idx, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Sequential processing due to dependencies
    # Launch one thread per element
    grid = (n_elements,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=1
    )