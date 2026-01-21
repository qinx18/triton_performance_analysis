import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a_val = tl.load(a_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    e_val = tl.load(e_ptr + offsets, mask=mask)
    
    # Control flow: if (a[i] > 0.) goto L20
    cond1 = a_val > 0.0
    
    # Path 1: a[i] <= 0.0
    # b[i] = -b[i] + d[i] * d[i]
    b_new = -b_val + d_val * d_val
    
    # if (b[i] <= a[i]) goto L30
    cond2 = b_new <= a_val
    
    # If not goto L30, execute: c[i] += d[i] * e[i]
    c_path1_update = c_val + d_val * e_val
    c_path1 = tl.where(cond2, c_val, c_path1_update)
    
    # Path 2: a[i] > 0.0 (L20)
    # c[i] = -c[i] + e[i] * e[i]
    c_path2 = -c_val + e_val * e_val
    
    # Select between paths
    b_final = tl.where(cond1, b_val, b_new)  # Only update b if a[i] <= 0
    c_final = tl.where(cond1, c_path2, c_path1)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_val
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)