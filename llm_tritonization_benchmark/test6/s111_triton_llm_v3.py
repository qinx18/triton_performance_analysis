import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets < n_elements) & ((current_offsets % 2) == 1)
        
        # Load a[i-1] and b[i] for valid odd indices
        a_prev_offsets = current_offsets - 1
        a_prev_mask = mask & (a_prev_offsets >= 0)
        b_mask = mask
        
        a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=b_mask, other=0.0)
        
        # Compute a[i] = a[i-1] + b[i]
        result = a_prev + b_vals
        
        # Store back to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s111_kernel[(1,)](a, b, n_elements, BLOCK_SIZE)