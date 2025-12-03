import triton
import triton.language as tl
import torch

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load arrays
    a_val = tl.load(a_ptr + current_offsets, mask=mask)
    b_val = tl.load(b_ptr + current_offsets, mask=mask)
    c_val = tl.load(c_ptr + current_offsets, mask=mask)
    d_val = tl.load(d_ptr + current_offsets, mask=mask)
    e_val = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Compute x = b[i]*c[i] + a[i]*d[i] + e[i]
    x_val = b_val * c_val + a_val * d_val + e_val
    
    # Store x value
    tl.store(x_ptr + current_offsets, x_val, mask=mask)
    
    # Compute and store a[i] = x - 1.0
    a_new = x_val - 1.0
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + current_offsets, x_val, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1281_kernel[grid](
        a, b, c, d, e, x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )