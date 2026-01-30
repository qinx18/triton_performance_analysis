import triton
import triton.language as tl
import torch

@triton.jit
def vpv_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        result = a_vals + b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def vpv_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    vpv_kernel[(1,)](
        a, b, N, BLOCK_SIZE
    )
    
    return a