import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must process sequentially due to b[i+1] = ... + b[i] dependency
    # Use single thread to maintain sequential dependency
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    for i in range(n_elements - 1):
        # Load current values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Check if a[i] >= 0 (goto L20)
        if a_val >= 0.0:
            # Skip to end (L20)
            pass
        else:
            # Check if b[i] >= 0 (goto L30)
            if b_val >= 0.0:
                # Only execute L30: b[i+1] = c[i] + d[i] * e[i]
                result = c_val + d_val * e_val
                tl.store(b_ptr + i + 1, result)
            else:
                # Execute both: a[i] += c[i] * d[i] and L30: b[i+1] = c[i] + d[i] * e[i]
                a_new = a_val + c_val * d_val
                tl.store(a_ptr + i, a_new)
                
                b_new = c_val + d_val * e_val
                tl.store(b_ptr + i + 1, b_new)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch single thread to maintain sequential dependency
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements
    )