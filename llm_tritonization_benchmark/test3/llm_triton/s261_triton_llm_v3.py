import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load c[i-1] values
    c_prev_offsets = offsets - 1
    c_prev_mask = (offsets >= 1) & mask
    c_prev_vals = tl.load(c_ptr + c_prev_offsets, mask=c_prev_mask)
    
    # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    a_new = t1 + c_prev_vals
    
    # Compute: t = c[i] * d[i]; c[i] = t
    t2 = c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(c_ptr + offsets, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # We process elements starting from index 1, so we need n_elements - 1 elements
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c