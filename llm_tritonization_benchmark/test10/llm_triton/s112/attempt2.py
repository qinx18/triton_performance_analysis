import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process all blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load from read-only copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = a_vals + b_vals
        
        # Store to a[i+1] positions
        store_offsets = current_offsets + 1
        tl.store(a_ptr + store_offsets, result)

def s112_triton(a, b):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to avoid WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE
    )