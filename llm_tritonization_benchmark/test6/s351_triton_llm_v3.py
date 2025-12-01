import triton
import triton.language as tl
import torch

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    for unroll_start in range(0, BLOCK_SIZE, 5):
        current_offsets = block_start + unroll_start + tl.arange(0, 5)
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        result = a_vals + alpha * b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s351_triton(a, b, c):
    alpha = c[0].item()
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s351_kernel[grid](a, b, alpha, n_elements, BLOCK_SIZE)
    
    return a