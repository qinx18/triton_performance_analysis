import triton
import triton.language as tl
import torch

@triton.jit
def s1279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # if (a[i] < 0.0)
    cond1 = a_vals < 0.0
    # if (b[i] > a[i])
    cond2 = b_vals > a_vals
    # Combined condition
    combined_cond = cond1 & cond2 & mask
    
    # c[i] += d[i] * e[i]
    update = d_vals * e_vals
    c_vals = tl.where(combined_cond, c_vals + update, c_vals)
    
    tl.store(c_ptr + indices, c_vals, mask=mask)

def s1279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )