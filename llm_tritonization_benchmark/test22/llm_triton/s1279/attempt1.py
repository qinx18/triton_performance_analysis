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
    
    # if (a[i] < 0.)
    a_negative = a_vals < 0.0
    
    # if (b[i] > a[i])
    b_greater_a = b_vals > a_vals
    
    # Combined condition: a[i] < 0 AND b[i] > a[i]
    update_mask = a_negative & b_greater_a & mask
    
    # c[i] += d[i] * e[i]
    update_vals = c_vals + d_vals * e_vals
    
    # Store only where condition is true
    tl.store(c_ptr + indices, update_vals, mask=update_mask)

def s1279_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )