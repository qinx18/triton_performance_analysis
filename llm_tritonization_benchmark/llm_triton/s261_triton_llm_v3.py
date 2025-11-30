import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Start from index 1, so adjust offsets
    offsets = offsets + 1
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_prev_vals = tl.load(c_ptr + offsets - 1, mask=mask)  # c[i-1]
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    new_a = t1 + c_prev_vals
    
    # Compute: t = c[i] * d[i]; c[i] = t
    t2 = c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(c_ptr + offsets, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # We process from index 1 to n_elements-1
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c