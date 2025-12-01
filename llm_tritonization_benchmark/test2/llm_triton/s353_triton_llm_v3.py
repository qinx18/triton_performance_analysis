import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the block, handling the unroll factor of 5
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Process 5 elements at a time within each thread's work
    for unroll_offset in range(0, BLOCK_SIZE, 5):
        current_offsets = block_start + unroll_offset + tl.arange(0, 5)
        current_mask = current_offsets < n_elements
        
        if tl.any(current_mask):
            # Load indices for gather operation
            ip_indices = tl.load(ip_ptr + current_offsets, mask=current_mask)
            
            # Load a values
            a_vals = tl.load(a_ptr + current_offsets, mask=current_mask)
            
            # Gather b values using ip indices
            b_vals = tl.load(b_ptr + ip_indices, mask=current_mask)
            
            # Compute saxpy operation
            result = a_vals + alpha * b_vals
            
            # Store results back to a
            tl.store(a_ptr + current_offsets, result, mask=current_mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    # Round up to handle the unroll factor of 5
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )