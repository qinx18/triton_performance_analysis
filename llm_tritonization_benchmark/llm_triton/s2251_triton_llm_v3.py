import triton
import triton.language as tl
import torch

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s = 0.0
    
    for i in range(n_elements):
        if i < n_elements:
            # Load current values
            e_val = tl.load(e_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + i, a_val)
            
            # Update s = b[i] + c[i]
            s = b_val + c_val
            
            # Compute b[i] = a[i] + d[i]
            b_new = a_val + d_val
            tl.store(b_ptr + i, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single program since this is inherently sequential
    BLOCK_SIZE = 1
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b