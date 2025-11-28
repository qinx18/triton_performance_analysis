import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Handle unrolled loop pattern - process 5 elements at a time
    for unroll_offset in range(5):
        current_offsets = offsets * 5 + unroll_offset
        current_mask = mask & (current_offsets < n_elements)
        
        # Load indices for gather operation
        ip_indices = tl.load(ip_ptr + current_offsets, mask=current_mask, other=0)
        
        # Gather from b array using loaded indices
        b_values = tl.load(b_ptr + ip_indices, mask=current_mask, other=0.0)
        
        # Load current a values
        a_values = tl.load(a_ptr + current_offsets, mask=current_mask, other=0.0)
        
        # Compute saxpy: a[i] += alpha * b[ip[i]]
        result = a_values + alpha * b_values
        
        # Store back to a
        tl.store(a_ptr + current_offsets, result, mask=current_mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0]
    
    # Calculate grid size for unrolled loop (process groups of 5)
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(triton.cdiv(n_elements, 5), BLOCK_SIZE)
    
    s353_kernel[(grid_size,)](
        a, b, ip, alpha, n_elements, BLOCK_SIZE
    )
    
    return a