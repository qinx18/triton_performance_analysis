import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel_1(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    result = b_vals + c_vals
    
    tl.store(a_ptr + offsets + 1, result, mask=mask)

@triton.jit
def s3251_kernel_2(b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    result = c_vals * e_vals
    
    tl.store(b_ptr + offsets, result, mask=mask)

@triton.jit
def s3251_kernel_3(a_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    result = a_vals * e_vals
    
    tl.store(d_ptr + offsets, result, mask=mask)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s3251_kernel_1[grid](a, b, c, n, BLOCK_SIZE)
    s3251_kernel_2[grid](a, c, e, n, BLOCK_SIZE)
    s3251_kernel_3[grid](a, d, e, n, BLOCK_SIZE)