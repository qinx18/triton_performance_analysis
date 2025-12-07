import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, strip_start: tl.constexpr, strip_size: tl.constexpr, n_elements: tl.constexpr):
    offsets = tl.arange(0, strip_size)
    idx = strip_start + offsets + 1  # +1 because original loop starts at i=1
    
    mask = idx < n_elements + 1
    
    # Load values
    b_prev = tl.load(b_ptr + idx - 1, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i-1] + c[i] * d[i]
    a_vals = b_prev + c_vals * d_vals
    
    # Compute b[i] = a[i] + c[i] * e[i]
    b_vals = a_vals + c_vals * e_vals
    
    # Store results
    tl.store(a_ptr + idx, a_vals, mask=mask)
    tl.store(b_ptr + idx, b_vals, mask=mask)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes from 1 to LEN_1D-1
    STRIP_SIZE = 1  # Must be 1 due to loop-carried dependency
    
    # Process strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        remaining = min(STRIP_SIZE, n_elements - strip_start)
        
        s323_kernel[(1,)](
            a, b, c, d, e,
            strip_start=strip_start,
            strip_size=remaining,
            n_elements=n_elements
        )