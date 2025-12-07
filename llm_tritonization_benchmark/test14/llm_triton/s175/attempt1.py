import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Check if this offset pattern matches the strided access
        stride_mask = (current_offsets % inc) == 0
        combined_mask = mask & stride_mask
        
        # Check bounds for read access (i + inc must be valid)
        read_offsets = current_offsets + inc
        read_mask = combined_mask & (read_offsets < (n_elements + 1))  # Original bound was LEN_1D-1, so read can go to LEN_1D
        
        # Load values where valid
        a_read_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=read_mask, other=0.0)
        
        # Compute result
        result = a_read_vals + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=read_mask)

def s175_triton(a, b, inc):
    n_elements = a.shape[0] - 1  # Loop bound is LEN_1D-1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n_elements, BLOCK_SIZE
    )