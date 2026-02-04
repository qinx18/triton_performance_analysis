import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel_loop1(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s244_kernel_loop2(b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    b_new = c_vals + b_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)

@triton.jit
def s244_kernel_loop3(a_ptr, b_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_plus1 = offsets + 1
    mask = offsets < n
    mask_plus1 = offsets_plus1 < (n + 1)
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals_plus1 = tl.load(a_ptr + offsets_plus1, mask=mask_plus1)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    a_new = b_vals + a_vals_plus1 * d_vals
    tl.store(a_ptr + offsets_plus1, a_new, mask=mask_plus1)

def s244_triton(a, b, c, d):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Loop 1: a[i] = b[i] + c[i] * d[i]
    s244_kernel_loop1[grid](a, b, c, d, n, BLOCK_SIZE)
    
    # Loop 2: b[i] = c[i] + b[i]
    s244_kernel_loop2[grid](b, c, n, BLOCK_SIZE)
    
    # Loop 3: a[i+1] = b[i] + a[i+1] * d[i] (only execute for last iteration)
    last_i = n - 1
    if last_i >= 0:
        grid_single = (1,)
        s244_kernel_loop3[grid_single](a, b, d, 1, 1)