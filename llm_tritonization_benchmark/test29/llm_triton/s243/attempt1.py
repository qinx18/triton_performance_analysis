import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel_loop1(a, b, c, d, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    a_vals = b_vals + c_vals * d_vals
    tl.store(a + offsets, a_vals, mask=mask)

@triton.jit
def s243_kernel_loop2(a, b, d, e, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    e_vals = tl.load(e + offsets, mask=mask)
    
    b_vals = a_vals + d_vals * e_vals
    tl.store(b + offsets, b_vals, mask=mask)

@triton.jit
def s243_kernel_loop3(a, a_copy, b, d, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    
    offsets_plus1 = offsets + 1
    mask_plus1 = offsets_plus1 < (n + 1)
    a_plus1_vals = tl.load(a_copy + offsets_plus1, mask=mask_plus1)
    
    a_vals = b_vals + a_plus1_vals * d_vals
    tl.store(a + offsets, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Create read-only copy for loop 3
    a_copy = a.clone()
    
    # Loop 1: a[i] = b[i] + c[i] * d[i]
    s243_kernel_loop1[grid](a, b, c, d, n, BLOCK_SIZE)
    
    # Loop 2: b[i] = a[i] + d[i] * e[i]
    s243_kernel_loop2[grid](a, b, d, e, n, BLOCK_SIZE)
    
    # Loop 3: a[i] = b[i] + a[i+1] * d[i]
    s243_kernel_loop3[grid](a, a_copy, b, d, n, BLOCK_SIZE)