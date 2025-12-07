import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Compute else branch condition
    else_mask = mask & (b_vals < 0.0)
    
    # c[i+1] = a[i] + d[i] * d[i]
    c_vals = a_vals + d_vals * d_vals
    tl.store(c_ptr + idx + 1, c_vals, mask=else_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute if branch condition  
    if_mask = mask & (b_vals >= 0.0)
    
    # a[i] = c[i] + d[i] * e[i]
    a_vals = c_vals + d_vals * e_vals
    tl.store(a_ptr + idx, a_vals, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process else branch first
    s161_kernel_else[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Process if branch after else completes
    s161_kernel_if[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )