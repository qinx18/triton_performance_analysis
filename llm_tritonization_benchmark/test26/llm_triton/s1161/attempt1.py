import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel_else(a_ptr, b_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    b_vals = a_vals + d_vals * d_vals
    
    tl.store(b_ptr + offsets, b_vals, mask=mask)

@triton.jit
def s1161_kernel_if(a_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    a_vals = c_vals + d_vals * e_vals
    
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process ALL 'else' branch iterations (c[i] < 0) FIRST
    else_mask = c[:n] < 0.0
    if else_mask.any():
        a_subset = a[:n][else_mask].contiguous()
        d_subset = d[:n][else_mask].contiguous()
        b_subset = b[:n][else_mask].contiguous()
        
        n_else = a_subset.shape[0]
        grid_else = (triton.cdiv(n_else, BLOCK_SIZE),)
        
        s1161_kernel_else[grid_else](
            a_subset, b_subset, d_subset, n_else, BLOCK_SIZE
        )
        
        # Copy results back
        b[:n][else_mask] = b_subset
    
    # Step 2: Process ALL 'if' branch iterations (c[i] >= 0) AFTER step 1 completes
    if_mask = c[:n] >= 0.0
    if if_mask.any():
        a_subset = a[:n][if_mask].contiguous()
        c_subset = c[:n][if_mask].contiguous()
        d_subset = d[:n][if_mask].contiguous()
        e_subset = e[:n][if_mask].contiguous()
        
        n_if = a_subset.shape[0]
        grid_if = (triton.cdiv(n_if, BLOCK_SIZE),)
        
        s1161_kernel_if[grid_if](
            a_subset, c_subset, d_subset, e_subset, n_if, BLOCK_SIZE
        )
        
        # Copy results back
        a[:n][if_mask] = a_subset