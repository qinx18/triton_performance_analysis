import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel_loop1(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Start from index 1, so add 1 to offsets
    indices = offsets + 1
    mask = indices < n_elements
    
    # Load data
    b_prev = tl.load(b_ptr + indices - 1, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute: a[i] = b[i-1] + c[i] * d[i]
    result = b_prev + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s323_kernel_loop2(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Start from index 1, so add 1 to offsets
    indices = offsets + 1
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute: b[i] = a[i] + c[i] * e[i]
    result = a_vals + c_vals * e_vals
    
    # Store result
    tl.store(b_ptr + indices, result, mask=mask)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for elements starting from index 1
    n_work = n_elements - 1
    grid = (triton.cdiv(n_work, BLOCK_SIZE),)
    
    # First loop: a[i] = b[i-1] + c[i] * d[i] for i in [1, n_elements)
    s323_kernel_loop1[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )
    
    # Second loop: b[i] = a[i] + c[i] * e[i] for i in [1, n_elements)
    s323_kernel_loop2[grid](
        a, b, c, e, n_elements, BLOCK_SIZE
    )