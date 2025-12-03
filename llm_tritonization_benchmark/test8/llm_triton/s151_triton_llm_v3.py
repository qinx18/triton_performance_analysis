import torch
import triton
import triton.language as tl

@triton.jit
def s151s_kernel(a_ptr, b_ptr, sign, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Subroutine logic: a[i] = sign * a[i] + b[0]
    b_val = tl.load(b_ptr)
    result = sign * a_vals + b_val
    
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s151_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    s151s_kernel(a_ptr, b_ptr, 1.0, n_elements, BLOCK_SIZE)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](a, b, n_elements, BLOCK_SIZE)