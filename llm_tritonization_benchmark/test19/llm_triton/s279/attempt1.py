import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # if (a[i] > 0.0)
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0.0
    b_vals_new = -b_vals + d_vals * d_vals
    
    # if (b[i] <= a[i]) after b update
    cond2 = b_vals_new <= a_vals
    
    # c[i] += d[i] * e[i] when both conditions false
    c_vals_path1 = c_vals + d_vals * e_vals
    
    # L20: c[i] = -c[i] + e[i] * e[i] when a[i] > 0.0
    c_vals_path2 = -c_vals + e_vals * e_vals
    
    # Select c values based on conditions
    # If a[i] > 0.0, use path2, otherwise check second condition
    c_vals_final = tl.where(cond1, c_vals_path2, 
                           tl.where(cond2, c_vals, c_vals_path1))
    
    # Update b values (only when a[i] <= 0.0)
    b_vals_final = tl.where(cond1, b_vals, b_vals_new)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_vals_final = b_vals_final + c_vals_final * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals_final, mask=mask)
    tl.store(b_ptr + offsets, b_vals_final, mask=mask)
    tl.store(c_ptr + offsets, c_vals_final, mask=mask)

def s279_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s279_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)