import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 1)
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    a_next_vals = tl.load(a_ptr + offsets + 1, mask=mask)
    
    # First computation: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second computation: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + offsets, b_new_vals, mask=mask)
    
    # Third computation: a[i+1] = b[i] + a[i+1] * d[i]
    a_next_new_vals = b_new_vals + a_next_vals * d_vals
    tl.store(a_ptr + offsets + 1, a_next_new_vals, mask=mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = ((n_elements - 1 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )