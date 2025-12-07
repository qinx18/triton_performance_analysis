import torch
import triton
import triton.language as tl

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load from read-only copy for a[i + inc]
        read_offsets = current_offsets + inc
        read_mask = mask & (read_offsets < n_elements + inc)
        a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = a[i + inc] + b[i]
        result = a_vals + b_vals
        
        # Store to original array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s175_triton(a, b, inc):
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Calculate number of elements to process
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s175_kernel[(grid_size,)](
        a, a_copy, b, inc, n_elements, BLOCK_SIZE
    )