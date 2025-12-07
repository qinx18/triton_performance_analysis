import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process one element at a time sequentially
    idx = tl.program_id(0)
    
    if idx >= n_elements:
        return
    
    # Initialize s based on previous iteration
    s = 0.0
    
    # Sequential computation for element idx
    for i in range(idx + 1):
        if i == 0:
            s = 0.0
        else:
            b_prev = tl.load(b_ptr + (i - 1))
            c_prev = tl.load(c_ptr + (i - 1))
            s = b_prev + c_prev
        
        if i == idx:
            # Load current values
            e_val = tl.load(e_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + i, a_val)
            
            # Compute b[i] = a[i] + d[i]
            b_new = a_val + d_val
            tl.store(b_ptr + i, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Launch one kernel per element due to sequential dependencies
    grid = (n_elements,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=1
    )