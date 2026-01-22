import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    a_plus_1_vals = tl.load(a_copy_ptr + offsets_plus_1, mask=mask_plus_1, other=0.0)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_vals = a_vals + d_vals * e_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_vals = b_vals + a_plus_1_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )