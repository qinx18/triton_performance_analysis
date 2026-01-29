import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    j = -1
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        for i in range(BLOCK_SIZE):
            actual_idx = block_start + i
            if actual_idx >= n_elements:
                return
                
            if mask[i] and b_vals[i] > 0.0:
                j = j + 1
                tl.store(a_ptr + j, b_vals[i])

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_elements))
    grid = (1,)
    
    s341_kernel[grid](
        b, a, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a