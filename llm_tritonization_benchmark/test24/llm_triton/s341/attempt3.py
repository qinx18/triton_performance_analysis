import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid > 0:
        return
    
    j = 0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_start = 0
    while block_start < n_elements:
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        i = 0
        while i < BLOCK_SIZE:
            element_idx = block_start + i
            if element_idx >= n_elements:
                return
            
            element_mask = i == offsets
            b_val = tl.sum(tl.where(element_mask, b_vals, 0.0))
            
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1
            
            i += 1
        
        block_start += BLOCK_SIZE

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s341_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)