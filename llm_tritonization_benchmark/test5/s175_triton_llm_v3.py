import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Check if current indices are valid (i.e., satisfy i % inc == 0)
        valid_indices = (current_offsets % inc) == 0
        combined_mask = mask & valid_indices
        
        # Load a[i + inc] and b[i]
        read_offsets = current_offsets + inc
        read_mask = combined_mask & (read_offsets < (n_elements + inc))
        
        a_inc_vals = tl.load(a_ptr + read_offsets, mask=read_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=combined_mask, other=0.0)
        
        # Compute a[i] = a[i + inc] + b[i]
        result = a_inc_vals + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=combined_mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    
    s175_kernel[(1,)](
        a, b, inc, n_elements, BLOCK_SIZE
    )