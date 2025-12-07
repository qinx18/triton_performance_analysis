import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Condition: a[i] > b[i]
    condition = a_vals > b_vals
    
    # Compute s = a[i] - b[i] * d[i] for elements where condition is true
    s = a_vals - b_vals * d_vals
    
    # Update c[i] += s and a[i] = s only where condition is true
    c_new = tl.where(condition, c_vals + s, c_vals)
    a_new = tl.where(condition, s, a_vals)
    
    # Store results
    tl.store(c_ptr + idx, c_new, mask=mask)
    tl.store(a_ptr + idx, a_new, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s253_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )