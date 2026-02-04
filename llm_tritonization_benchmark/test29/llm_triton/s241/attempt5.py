import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel_1(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    result = b_vals * c_vals * d_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s241_kernel_2(b_ptr, a_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    mask_next = (indices + 1) < (n_elements + 1)
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    a_next_vals = tl.load(a_ptr + indices + 1, mask=mask_next)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    result = a_vals * a_next_vals * d_vals
    
    tl.store(b_ptr + indices, result, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First kernel: a[i] = b[i] * c[i] * d[i]
    s241_kernel_1[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: b[i] = a[i] * a[i+1] * d[i]
    s241_kernel_2[grid](
        b, a, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )