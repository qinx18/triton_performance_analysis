import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets + 1  # Start from index 1
    
    mask = idx < n_elements
    
    # Read b[i-1]
    b_prev = tl.load(b_ptr + idx - 1, mask=mask)
    
    # Read c[i], d[i], e[i]
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
    n_elements = a.shape[0]
    elements_to_process = n_elements - 1  # Start from index 1
    
    BLOCK_SIZE = 1  # Must be 1 due to loop-carried dependency
    
    # Process elements sequentially due to b[i] dependency on b[i-1]
    for strip_start in range(0, elements_to_process, BLOCK_SIZE):
        grid = (1,)
        s323_kernel[grid](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )