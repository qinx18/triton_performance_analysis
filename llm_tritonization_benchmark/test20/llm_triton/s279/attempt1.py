import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # if (a[i] > 0.) goto L20
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0
    # b[i] = -b[i] + d[i] * d[i]
    b_new = -b_vals + d_vals * d_vals
    
    # if (b[i] <= a[i]) goto L30
    cond2 = b_new <= a_vals
    
    # When not going to L30: c[i] += d[i] * e[i]
    c_branch1 = c_vals + d_vals * e_vals
    
    # L20: c[i] = -c[i] + e[i] * e[i] (when a[i] > 0)
    c_branch2 = -c_vals + e_vals * e_vals
    
    # Select c value based on conditions
    # If a[i] > 0: use c_branch2
    # If a[i] <= 0 and b_new <= a[i]: use original c_vals
    # If a[i] <= 0 and b_new > a[i]: use c_branch1
    c_final = tl.where(cond1, c_branch2, 
                      tl.where(cond2, c_vals, c_branch1))
    
    # Select b value: if a[i] > 0, use original b_vals, else use b_new
    b_final = tl.where(cond1, b_vals, b_new)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)