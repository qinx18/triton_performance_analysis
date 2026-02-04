import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel_s0(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    result = b_vals + c_vals
    
    tl.store(a_ptr + offsets + 1, result, mask=mask)

@triton.jit
def s3251_kernel_s1(b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    result = c_vals * e_vals
    
    tl.store(b_ptr + offsets, result, mask=mask)

@triton.jit
def s3251_kernel_s2(d_ptr, a_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    result = a_vals * e_vals
    
    tl.store(d_ptr + offsets, result, mask=mask)

def s3251_triton(a, b, c, d, e):
    N = a.shape[0]
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # S0: a[i+1] = b[i] + c[i]
    s3251_kernel_s0[grid](a, b, c, n_elements, BLOCK_SIZE)
    
    # S1: b[i] = c[i] * e[i]
    s3251_kernel_s1[grid](b, c, e, n_elements, BLOCK_SIZE)
    
    # S2: d[i] = a[i] * e[i]
    s3251_kernel_s2[grid](d, a, e, n_elements, BLOCK_SIZE)