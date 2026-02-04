import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel_loop1(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust for starting from index 1
    indices = offsets + 1
    mask = indices < n_elements
    
    # Load b[i-1], c[i], d[i]
    b_prev_indices = indices - 1
    b_vals = tl.load(b_ptr + b_prev_indices, mask=mask)  # b[i-1]
    c_vals = tl.load(c_ptr + indices, mask=mask)  # c[i]
    d_vals = tl.load(d_ptr + indices, mask=mask)  # d[i]
    
    # Compute a[i] = b[i-1] + c[i] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store a[i]
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s323_kernel_loop2(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Adjust for starting from index 1
    indices = offsets + 1
    mask = indices < n_elements
    
    # Load a[i], c[i], e[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)  # a[i]
    c_vals = tl.load(c_ptr + indices, mask=mask)  # c[i]
    e_vals = tl.load(e_ptr + indices, mask=mask)  # e[i]
    
    # Compute b[i] = a[i] + c[i] * e[i]
    result = a_vals + c_vals * e_vals
    
    # Store b[i]
    tl.store(b_ptr + indices, result, mask=mask)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # We process from index 1 to n_elements-1
    n_work = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_work, BLOCK_SIZE),)
    
    # Loop 1: a[i] = b[i-1] + c[i] * d[i]
    s323_kernel_loop1[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    # Loop 2: b[i] = a[i] + c[i] * e[i]
    s323_kernel_loop2[grid](a, b, c, e, n_elements, BLOCK_SIZE)