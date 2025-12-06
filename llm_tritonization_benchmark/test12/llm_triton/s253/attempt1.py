import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Condition: a[i] > b[i]
    condition = a_vals > b_vals
    
    # s = a[i] - b[i] * d[i]
    s_vals = a_vals - b_vals * d_vals
    
    # c[i] += s (only where condition is true)
    new_c = tl.where(condition, c_vals + s_vals, c_vals)
    
    # a[i] = s (only where condition is true)
    new_a = tl.where(condition, s_vals, a_vals)
    
    tl.store(c_ptr + indices, new_c, mask=mask)
    tl.store(a_ptr + indices, new_a, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s253_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )