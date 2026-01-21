import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    b_mask = mask & (tl.load(b_ptr + offsets, mask=mask, other=1.0) < 0.0)
    
    a_vals = tl.load(a_ptr + offsets, mask=b_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=b_mask, other=0.0)
    
    c_out = a_vals + d_vals * d_vals
    tl.store(c_ptr + offsets + 1, c_out, mask=b_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    b_mask = mask & (tl.load(b_ptr + offsets, mask=mask, other=-1.0) >= 0.0)
    
    c_vals = tl.load(c_ptr + offsets, mask=b_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=b_mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=b_mask, other=0.0)
    
    a_out = c_vals + d_vals * e_vals
    tl.store(a_ptr + offsets, a_out, mask=b_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process else branch (b[i] < 0) first
    s161_kernel_else[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Process if branch (b[i] >= 0) after else completes
    s161_kernel_if[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)