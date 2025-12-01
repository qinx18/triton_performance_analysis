import triton
import triton.language as tl
import torch

@triton.jit
def vpvtv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        result = a_vals + b_vals * c_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def vpvtv_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    vpvtv_kernel[(1,)](a, b, c, n_elements, BLOCK_SIZE)