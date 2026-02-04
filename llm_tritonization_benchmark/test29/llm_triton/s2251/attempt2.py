import triton
import triton.language as tl
import torch

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    e_vals = tl.load(e_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Calculate s values using scalar expansion
    # s = 0.0 for i=0, s = b[i-1] + c[i-1] for i>0
    s_vals = tl.zeros_like(e_vals)
    
    # For i > 0, s = b[i-1] + c[i-1]
    prev_indices = indices - 1
    prev_mask = mask & (indices > 0)
    
    b_prev = tl.load(b_ptr + prev_indices, mask=prev_mask)
    c_prev = tl.load(c_ptr + prev_indices, mask=prev_mask)
    s_vals = tl.where(prev_mask, b_prev + c_prev, s_vals)
    
    # Compute a[i] = s * e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i] + d[i]
    b_new = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)
    tl.store(b_ptr + indices, b_new, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )