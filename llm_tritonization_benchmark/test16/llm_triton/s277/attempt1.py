import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr):
    # This must be sequential due to the b[i+1] = ... b[i] dependency
    # Use single thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all elements sequentially in a single thread
    for i in range(n_elements - 1):
        # Load current values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Check first condition: if a[i] >= 0, skip everything (goto L20)
        a_ge_zero = a_val >= 0.0
        
        # Check second condition: if b[i] >= 0, skip to L30
        b_ge_zero = b_val >= 0.0
        
        # If a[i] < 0 and b[i] < 0, execute: a[i] += c[i] * d[i]
        should_update_a = (a_val < 0.0) & (b_val < 0.0)
        new_a = tl.where(should_update_a, a_val + c_val * d_val, a_val)
        tl.store(a_ptr + i, new_a)
        
        # L30: Execute b[i+1] = c[i] + d[i] * e[i] if a[i] < 0
        should_update_b = a_val < 0.0
        new_b = c_val + d_val * e_val
        if should_update_b:
            tl.store(b_ptr + i + 1, new_b)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single thread since this must be sequential
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements=n_elements
    )
    
    return a, b