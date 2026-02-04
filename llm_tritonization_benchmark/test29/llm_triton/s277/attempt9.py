import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    i = 0
    while i < n_elements - 1:
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Check if a[i] >= 0 - if true, skip everything (goto L20)
        if a_val >= 0.0:
            i += 1
        else:
            # Load b[i]
            b_val = tl.load(b_ptr + i)
            
            # Check if b[i] >= 0 - if true, skip a update but do L30
            if b_val < 0.0:
                # Update a[i]: a[i] += c[i] * d[i]
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                a_new = a_val + c_val * d_val
                tl.store(a_ptr + i, a_new)
            
            # L30: Always execute when we didn't skip everything
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            b_new = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, b_new)
            i += 1

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single block for sequential execution
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e, 
        n_elements
    )