import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_step1(a_ptr, c_ptr, d_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    else_mask = mask & (b_vals < 0.0)
    
    a_vals = tl.load(a_ptr + offsets, mask=else_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=else_mask, other=0.0)
    result = a_vals + d_vals * d_vals
    
    output_offsets = offsets + 1
    output_mask = else_mask & (output_offsets <= n)
    tl.store(c_ptr + output_offsets, result, mask=output_mask)

@triton.jit  
def s161_kernel_step2(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    if_mask = mask & (b_vals >= 0.0)
    
    c_vals = tl.load(c_ptr + offsets, mask=if_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=if_mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=if_mask, other=0.0)
    result = c_vals + d_vals * e_vals
    
    tl.store(a_ptr + offsets, result, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process else branch (c[i+1] = a[i] + d[i] * d[i])
    s161_kernel_step1[grid](a, c, d, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Process if branch (a[i] = c[i] + d[i] * e[i])  
    s161_kernel_step2[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)