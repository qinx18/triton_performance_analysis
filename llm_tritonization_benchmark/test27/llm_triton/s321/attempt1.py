import triton
import triton.language as tl
import torch

@triton.jit
def s321_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Start from i=1 since we need a[i-1]
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Load previous values a[i-1]
        prev_offsets = current_offsets - 1
        prev_mask = (prev_offsets >= 0) & mask
        a_prev_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask)
        
        # Compute a[i] += a[i-1] * b[i]
        result = a_vals + a_prev_vals * b_vals
        
        # Store back
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s321_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential processing due to recurrence dependency
    for i in range(1, n_elements):
        a[i] += a[i-1] * b[i]