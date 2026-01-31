import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    else_mask = (b_vals < 0.0) & mask
    
    a_vals = tl.load(a_ptr + offsets, mask=else_mask)
    d_vals = tl.load(d_ptr + offsets, mask=else_mask)
    
    result = a_vals + d_vals * d_vals
    
    tl.store(c_ptr + offsets + 1, result, mask=else_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    if_mask = (b_vals >= 0.0) & mask
    
    c_vals = tl.load(c_ptr + offsets, mask=if_mask)
    d_vals = tl.load(d_ptr + offsets, mask=if_mask)
    e_vals = tl.load(e_ptr + offsets, mask=if_mask)
    
    result = c_vals + d_vals * e_vals
    
    tl.store(a_ptr + offsets, result, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process else branch first
    s161_kernel_else[grid](
        a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Process if branch after else branch completes
    s161_kernel_if[grid](
        a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE
    )