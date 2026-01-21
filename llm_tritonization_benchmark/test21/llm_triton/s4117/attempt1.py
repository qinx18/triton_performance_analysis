import triton
import triton.language as tl
import torch

@triton.jit
def s4117_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    c_idx = idx // 2
    c_mask = c_idx < n_elements
    c_vals = tl.load(c_ptr + c_idx, mask=mask & c_mask)
    
    result = b_vals + c_vals * d_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s4117_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4117_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)