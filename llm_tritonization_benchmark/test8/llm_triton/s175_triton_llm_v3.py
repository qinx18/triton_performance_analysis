import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, b_ptr, n_elements, inc, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load a[i + inc] and b[i]
        a_inc_offsets = current_offsets + inc
        a_inc_mask = mask & (a_inc_offsets < (n_elements + inc))
        
        a_inc_vals = tl.load(a_ptr + a_inc_offsets, mask=a_inc_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = a[i + inc] + b[i]
        result = a_inc_vals + b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    grid = (1,)
    s175_kernel[grid](a, b, n_elements, inc, BLOCK_SIZE)