import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load from read-only copy and b array
        a_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Compute result
        result = a_vals + b_vals
        
        # Write to original array at offset + 1
        write_offsets = current_offsets + 1
        write_mask = write_offsets < n_elements
        tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b):
    n_elements = a.shape[0] - 1  # Process elements 0 to LEN_1D-2
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )