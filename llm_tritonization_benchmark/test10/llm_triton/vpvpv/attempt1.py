import triton
import triton.language as tl
import torch

@triton.jit
def vpvpv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    a_ptrs = a_ptr + block_start + offsets
    b_ptrs = b_ptr + block_start + offsets
    c_ptrs = c_ptr + block_start + offsets
    
    a_vals = tl.load(a_ptrs, mask=mask)
    b_vals = tl.load(b_ptrs, mask=mask)
    c_vals = tl.load(c_ptrs, mask=mask)
    
    result = a_vals + b_vals + c_vals
    
    tl.store(a_ptrs, result, mask=mask)

def vpvpv_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vpvpv_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)