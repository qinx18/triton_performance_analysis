import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        result = b_vals + 1.0
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s1112_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1112_kernel[grid](a, b, N, BLOCK_SIZE)