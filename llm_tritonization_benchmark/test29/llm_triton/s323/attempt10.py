import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel_loop1(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    indices = offsets + 1
    mask = indices < n_elements
    
    b_prev_indices = indices - 1
    b_vals = tl.load(b_ptr + b_prev_indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    result = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, result, mask=mask)

@triton.jit
def s323_kernel_loop2(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    indices = offsets + 1
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    result = a_vals + c_vals * e_vals
    tl.store(b_ptr + indices, result, mask=mask)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    num_elements_to_process = n_elements - 1
    grid = (triton.cdiv(num_elements_to_process, BLOCK_SIZE),)
    
    s323_kernel_loop1[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    s323_kernel_loop2[grid](a, b, c, e, n_elements, BLOCK_SIZE)