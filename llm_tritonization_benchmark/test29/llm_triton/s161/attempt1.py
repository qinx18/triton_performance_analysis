import triton
import triton.language as tl
import torch

@triton.jit
def s161_else_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    else_mask = mask & (b_vals < 0.0)
    
    a_vals = tl.load(a_ptr + idx, mask=else_mask)
    d_vals = tl.load(d_ptr + idx, mask=else_mask)
    
    c_new = a_vals + d_vals * d_vals
    
    tl.store(c_ptr + idx + 1, c_new, mask=else_mask)

@triton.jit
def s161_if_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    if_mask = mask & (b_vals >= 0.0)
    
    c_vals = tl.load(c_ptr + idx, mask=if_mask)
    d_vals = tl.load(d_ptr + idx, mask=if_mask)
    e_vals = tl.load(e_ptr + idx, mask=if_mask)
    
    a_new = c_vals + d_vals * e_vals
    
    tl.store(a_ptr + idx, a_new, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process else branch first
    s161_else_kernel[grid](
        a, b, c, d,
        n,
        BLOCK_SIZE,
    )
    
    # Step 2: Process if branch after else branch completes
    s161_if_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE,
    )